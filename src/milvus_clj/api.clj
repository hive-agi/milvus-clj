(ns milvus-clj.api
  "Legacy thin compat shim over `milvus-clj.client`.

   Historical shape: a flat API of 21 functions holding a `defonce`
   singleton MilvusServiceClient. The protocol refactor (see
   `milvus-clj.client`) split the implementation into transport-
   specific records under `milvus-clj.transport.*`. This ns survives
   as a delegating layer so existing callers (hive-milvus is the sole
   consumer) need not migrate in one PR.

   The singleton lives in `*default-client*`. `connect!` builds a
   `GrpcClient` via `client/make` and stores it. Every op reads
   `@default-client`, dispatches to the protocol method, and wraps
   the return in `future` to preserve the legacy `@(milvus/...)`
   call pattern hive-milvus depends on.

   This ns will be deleted in PR-4 once all call sites migrate to
   `milvus-clj.client/with-client`.

   Backward compatibility notes:
     - `connect!` / `disconnect!` / `connected?` preserve the old
       singleton lifecycle.
     - Every op still returns a future (deref with `@`).
     - Config keys merged via `connect!` pass through to
       `milvus-clj.client/make` under `{:transport :grpc}` defaults.
     - `resolve-consistency` + `with-connection` macro preserved
       (hive-milvus uses neither today, but external callers might)."
  (:refer-clojure :exclude [get])
  (:require [milvus-clj.config :as config]
            [milvus-clj.client :as client]
            [milvus-clj.transport.grpc :as grpc]))

;; ============================================================================
;; Singleton lifecycle
;; ============================================================================

(defonce ^:private default-client (atom nil))

(defn- require-client!
  "Return the live client or throw — mirrors the historical error msg."
  []
  (or @default-client
      (throw (ex-info "Milvus client not connected. Call (connect!) first." {}))))

(defn connect!
  "Connect to Milvus using the gRPC transport. Preserved signature:
     (connect!)
     (connect! {:host \"milvus.prod\" :port 19530})
   Merges opts into `milvus-clj.config` (legacy flat shape) and
   constructs a `GrpcClient` via `milvus-clj.client/make`. Returns the
   underlying `MilvusServiceClient` for backward compat — note that
   callers holding the old raw client reference will still work, but
   new code should use `milvus-clj.client/with-client` + protocol
   methods instead."
  ([] (connect! {}))
  ([opts]
   (when (seq opts) (config/configure! opts))
   (let [flat (config/get-config)
         ;; Map the legacy flat config onto the new nested shape.
         nested {:transport :grpc
                 :host      (:host flat)
                 :port      (:port flat)
                 :token     (:token flat)
                 :database  (:database flat)
                 :grpc {:connect-timeout-ms        (:connect-timeout-ms flat)
                        :keep-alive-time-ms        (:keep-alive-time-ms flat)
                        :keep-alive-timeout-ms     (:keep-alive-timeout-ms flat)
                        :keep-alive-without-calls? (:keep-alive-without-calls? flat)
                        :idle-timeout-ms           (:idle-timeout-ms flat)
                        :secure                    (:secure flat)}}
         gc   ^milvus_clj.transport.grpc.GrpcClient (client/make nested)]
     (reset! default-client gc)
     ;; Legacy callers expect the raw MilvusServiceClient.
     (:msc gc))))

(defn disconnect!
  "Close the active client and clear the singleton. Idempotent."
  []
  (when-let [gc @default-client]
    (client/-close gc)
    (reset! default-client nil)))

(defn connected?
  "Check if the singleton client is set."
  []
  (some? @default-client))

(defmacro with-connection
  "Execute body with a temporary Milvus connection that auto-disconnects.
   Preserved for backward compat — new code should use
   `milvus-clj.client/with-client`."
  [opts & body]
  `(try
     (connect! ~opts)
     ~@body
     (finally
       (disconnect!))))

;; ============================================================================
;; Consistency (re-exported from transport/grpc so external callers keep
;; ============================================================================

(def consistency-levels
  "Map keyword → ConsistencyLevelEnum (gRPC-only concept)."
  grpc/consistency-levels)

(defn resolve-consistency
  "Resolve keyword to a gRPC ConsistencyLevelEnum. Nil means default."
  [kw]
  (grpc/resolve-consistency kw))

;; ============================================================================
;; Collection management — delegates to IMilvusAdmin + IMilvusCore + IMilvusExtras
;; ============================================================================

(defn create-collection
  "Create a Milvus collection with schema + optional index. Returns future.
   See `milvus-clj.client/-create-collection`."
  [collection-name opts]
  (future (client/-create-collection (require-client!) collection-name opts)))

(defn has-collection
  "Check if a collection exists. Returns future<boolean>."
  [collection-name]
  (future (client/-has-collection (require-client!) collection-name)))

(defn get-collection
  "Describe a collection's schema. Returns future<map>."
  [collection-name]
  (future (client/-describe (require-client!) collection-name)))

(defn list-collections
  "List all collection names. Returns future<vector<string>>."
  []
  (future (client/-list-collections (require-client!))))

(defn drop-collection
  "Drop a collection. Returns future."
  [collection-name]
  (future (client/-drop-collection (require-client!) collection-name)))

(defn load-collection
  "Load a collection into memory. Returns future."
  [collection-name]
  (future (client/-load-collection (require-client!) collection-name)))

(defn release-collection
  "Release a collection from memory. Returns future."
  [collection-name]
  (future (client/-release-collection (require-client!) collection-name)))

;; ============================================================================
;; Data operations (CRUD) — delegates to IMilvusCore
;; ============================================================================

(defn add
  "Insert or upsert records. Returns future.
   Options: :upsert? (default false), :partition."
  [collection-name records & {:keys [upsert? partition]
                              :or {upsert? false}}]
  (let [opts {:partition partition}
        cl   (require-client!)]
    (future
      (if upsert?
        (client/-upsert cl collection-name records opts)
        (client/-insert cl collection-name records opts)))))

(defn get
  "Get records by IDs. Returns future<vec>.
   Options: :include, :consistency-level."
  [collection-name ids & {:keys [include consistency-level]
                          :or {consistency-level :strong}}]
  (let [opts {:include include :consistency-level consistency-level}]
    (future (client/-get (require-client!) collection-name ids opts))))

(defn delete
  "Delete records by IDs. Returns future.
   Options: :partition."
  [collection-name ids & {:keys [partition]}]
  (let [opts {:partition partition}]
    (future (client/-delete (require-client!) collection-name ids opts))))

;; ============================================================================
;; Search / Query
;; ============================================================================

(defn query
  "Vector similarity search. Returns future<vec>.
   See `milvus-clj.client/-query` for q-map shape."
  [collection-name q-map]
  (future (client/-query (require-client!) collection-name q-map)))

(defn query-scalar
  "Scalar filter query. Returns future<vec>."
  [collection-name q-map]
  (future (client/-query-scalar (require-client!) collection-name q-map)))

;; ============================================================================
;; Convenience / extras
;; ============================================================================

(defn flush-collection
  "Flush collection to storage. Returns future."
  [collection-name]
  (future (client/-flush-collection (require-client!) collection-name)))

(defn collection-stats
  "Basic collection stats. Returns future<map>."
  [collection-name]
  (future
    (try
      {:exists? @(has-collection collection-name)
       :info    @(get-collection collection-name)}
      (catch Exception e
        {:error (str e)}))))

(defn create-index
  "Create an index on a collection field. Returns future."
  [collection-name index-opts]
  (future (client/-create-index (require-client!) collection-name index-opts)))

(defn drop-index
  "Drop an index from a collection field. Returns future."
  [collection-name field-name]
  (future (client/-drop-index (require-client!) collection-name field-name)))
