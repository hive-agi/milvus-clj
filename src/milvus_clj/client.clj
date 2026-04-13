(ns milvus-clj.client
  "Transport-abstract Milvus client.

   Defines the `IMilvusCore`, `IMilvusAdmin`, `IMilvusExtras` protocols,
   the `make` factory dispatching on `:transport`, the `with-client`
   macro for ephemeral lifetimes, and `classify-error` for transport-
   specific failure classification.

   Caller picks the transport:

       (require '[milvus-clj.client :as client])

       (client/with-client [c {:transport :grpc
                               :host \"milvus.prod\" :port 19530
                               :token \"root:pw\"}]
         (client/-query c \"memories\" {:vector [...] :limit 10}))

       (client/with-client [c {:transport :http
                               :host \"milvus.prod\" :port 9091
                               :token \"root:pw\"}]
         (client/-query c \"memories\" {:vector [...] :limit 10}))

   Adding a third transport is a new file under `milvus-clj.transport.*`
   plus one extra case arm in `make`. Nothing else changes — OCP.")

;; ============================================================================
;; Protocols — ISP split
;; ============================================================================

(defprotocol IMilvusCore
  "Core data-plane operations. Required for every transport.
   All methods are synchronous; the caller wraps in `future` if async
   is needed (the legacy `milvus-clj.api` shim does this for backward
   compat with hive-milvus's `@(milvus/...)` pattern)."
  (-has-collection  [client collection-name]
    "Return boolean — does the collection exist?")
  (-load-collection [client collection-name]
    "Load collection into memory for search. Idempotent.
     Returns `{:collection-name name :status :loaded}`.")
  (-insert       [client collection-name records opts]
    "Insert rows. Returns `{:count n :mutation :insert}`.")
  (-upsert       [client collection-name records opts]
    "Upsert rows (replace by PK). Returns `{:count n :mutation :upsert}`.")
  (-delete       [client collection-name ids opts]
    "Delete rows by ID list. Returns `{:deleted n}`.")
  (-get          [client collection-name ids opts]
    "Fetch rows by ID list. Returns a vector of entry maps.")
  (-query        [client collection-name q-map]
    "Vector similarity search. Returns a vector of entries each with
     `:distance` attached.")
  (-query-scalar [client collection-name q-map]
    "Scalar filter query (no vector). Returns a vector of entry maps.")
  (-close        [client]
    "Release resources. Idempotent."))

(defprotocol IMilvusAdmin
  "Collection admin operations. Implemented by gRPC and HTTP transports;
   optional for future specialized transports."
  (-create-collection [client collection-name opts]
    "Create a collection with schema + optional index config in one call.
     Returns `{:collection-name name :status :created}`.")
  (-drop-collection   [client collection-name]
    "Destructively drop the collection. Returns `{:collection-name name :status :dropped}`.")
  (-describe          [client collection-name]
    "Describe schema and load state. Doubles as a health probe."))

(defprotocol IMilvusExtras
  "Rarely-used operations. Optional — a minimal transport implementation
   may leave these unimplemented. hive-milvus does not call any of these
   today; kept in the protocol surface for future use."
  (-list-collections   [client])
  (-release-collection [client collection-name])
  (-flush-collection   [client collection-name])
  (-create-index       [client collection-name index-opts])
  (-drop-index         [client collection-name field-name]))

;; ============================================================================
;; Factory
;; ============================================================================

(defn make
  "Construct an `IMilvusCore` implementation for the chosen transport.

   `opts` must have `:transport` — one of `:grpc` or `:http`.
   Transport-specific keys live under `(:grpc opts)` or `(:http opts)`;
   agnostic keys (`:host`, `:port`, `:token`, `:database`) sit at the
   top level of `opts`.

   See `milvus-clj.config` for default merging and validation."
  [{:keys [transport] :as opts}]
  (case transport
    :grpc (let [grpc-ns (requiring-resolve 'milvus-clj.transport.grpc/open)]
            (grpc-ns opts))
    :http (let [http-ns (requiring-resolve 'milvus-clj.transport.http/open)]
            (http-ns opts))
    (throw (ex-info (str "Unknown :transport " (pr-str transport)
                         " — expected :grpc or :http")
                    {:transport transport
                     :supported #{:grpc :http}}))))

(defmacro with-client
  "Open a client, run body, close deterministically.

   Example:
     (with-client [c {:transport :grpc :host \"localhost\" :port 19530}]
       (-query c \"memories\" {:vector [...] :limit 10}))"
  [[binding opts] & body]
  `(let [~binding (make ~opts)]
     (try
       ~@body
       (finally
         (try (-close ~binding) (catch Throwable _# nil))))))

;; ============================================================================
;; Error classification
;; ============================================================================

(defn classify-error
  "Classify an exception thrown from a protocol method into one of:
     :connection-failure  — transport-level, safe to retry after reconnect
     :retryable           — transient (5xx, throttle), safe to retry in place
     :fatal               — caller bug / schema mismatch / auth failure

   Each transport attaches `::transport` in the ex-data. This dispatches
   on that key so the classification logic lives in the transport ns
   (only the transport knows which of its exception shapes are transient).

   Callers (e.g. `hive-milvus.store/with-auto-reconnect`) should use this
   predicate rather than pattern-matching on exception classes directly."
  [^Throwable ex]
  (let [data (ex-data ex)
        transport (::transport data)]
    (case transport
      :grpc (let [f (requiring-resolve 'milvus-clj.transport.grpc/classify)]
              (f ex))
      :http (let [f (requiring-resolve 'milvus-clj.transport.http/classify)]
              (f ex))
      ;; Unknown / no transport tag — try both classifiers in order.
      (or (try ((requiring-resolve 'milvus-clj.transport.grpc/classify) ex)
               (catch Throwable _ nil))
          :fatal))))
