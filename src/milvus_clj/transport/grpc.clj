(ns milvus-clj.transport.grpc
  "gRPC transport for `milvus-clj.client`. Wraps `io.milvus/milvus-sdk-java`.

   Responsibilities:
     - Build a MilvusServiceClient from config (the `open` fn).
     - Implement `IMilvusCore` + `IMilvusAdmin` + `IMilvusExtras` via a
       `GrpcClient` defrecord that holds the service client and its own
       config for lifecycle.
     - Translate ex-info-tagged errors for `client/classify-error`.

   Keepalive defaults are tuned for fragile intermediaries (tailscale
   userspace netstack, cloud NAT). `keepAliveWithoutCalls(true)` is the
   critical bit: without it gRPC only pings during active RPCs, so idle
   clients die silently with
   `UNAVAILABLE: Keepalive failed. The connection is likely gone`.

   This ns is loaded lazily via `requiring-resolve` from
   `milvus-clj.client/make`, so the gRPC Java SDK classes are only
   loaded when the gRPC transport is actually selected."
  (:require [milvus-clj.client :as client]
            [milvus-clj.schema :as schema]
            [milvus-clj.index :as idx]
            [clojure.data.json :as json]
            [clojure.string :as str]
            [taoensso.timbre :as log])
  (:import [io.milvus.client MilvusServiceClient]
           [io.milvus.param ConnectParam R]
           [io.milvus.param.collection
            CreateCollectionParam DropCollectionParam HasCollectionParam
            LoadCollectionParam ReleaseCollectionParam
            ShowCollectionsParam DescribeCollectionParam
            FlushParam]
           [io.milvus.param.dml
            InsertParam InsertParam$Field UpsertParam
            DeleteParam SearchParam QueryParam]
           [io.milvus.response
            SearchResultsWrapper QueryResultsWrapper
            DescCollResponseWrapper]
           [io.milvus.common.clientenum ConsistencyLevelEnum]
           [io.grpc StatusRuntimeException]
           [java.util ArrayList]
           [java.util.concurrent TimeUnit]))

;; ============================================================================
;; Helpers (unchanged from legacy api.clj — behaviour-preserving move)
;; ============================================================================

(def consistency-levels
  "Map keyword → Milvus ConsistencyLevelEnum.
   Public so `milvus-clj.api` can re-export for backward compat."
  {:strong     ConsistencyLevelEnum/STRONG
   :bounded    ConsistencyLevelEnum/BOUNDED
   :session    ConsistencyLevelEnum/SESSION
   :eventually ConsistencyLevelEnum/EVENTUALLY})

(defn resolve-consistency
  "Resolve keyword to ConsistencyLevelEnum. Nil means use collection default."
  [kw]
  (when kw (get consistency-levels kw ConsistencyLevelEnum/STRONG)))

(defn- check-response!
  "Validate a Milvus R<T> response. Returns data on success, throws on error.
   The Milvus Java SDK wraps all responses in R<T> with status codes."
  [^R response operation]
  (if (zero? (.getStatus response))
    (.getData response)
    (throw (ex-info (str "Milvus " operation " failed: " (.getMessage response))
                    {::client/transport :grpc
                     :status    (.getStatus response)
                     :message   (.getMessage response)
                     :exception (.getException response)
                     :operation operation}))))

(defn- rows->columns
  "Convert row-oriented data (vector of maps) to columnar format.
   Milvus SDK v1 requires columnar insertion via InsertParam.Field."
  [rows]
  (when (seq rows)
    (let [field-names (keys (first rows))]
      (reduce (fn [acc field]
                (assoc acc field (mapv #(get % field) rows)))
              {}
              field-names))))

(defn- coerce-embedding
  "Ensure embedding vector contains Java Float values for Milvus gRPC."
  [v]
  (when v (ArrayList. ^java.util.Collection (mapv float v))))

(defn- ->insert-fields
  "Convert columnar data map to list of InsertParam.Field objects.
   Handles embedding coercion (vectors must be List<Float>)."
  [columns]
  (ArrayList.
   ^java.util.Collection
   (mapv (fn [[field-name values]]
           (let [fname (name field-name)
                 vals  (if (= fname "embedding")
                         (ArrayList. ^java.util.Collection (mapv coerce-embedding values))
                         (ArrayList. ^java.util.Collection values))]
             (InsertParam$Field. fname vals)))
         columns)))

;; ============================================================================
;; Core operation bodies (protocol methods call these)
;; ============================================================================

(defn- do-has-collection
  [^MilvusServiceClient msc collection-name]
  (let [param (-> (HasCollectionParam/newBuilder)
                  (.withCollectionName collection-name)
                  .build)]
    (check-response! (.hasCollection msc param) "hasCollection")))

(defn- do-load-collection
  [^MilvusServiceClient msc collection-name]
  (let [param (-> (LoadCollectionParam/newBuilder)
                  (.withCollectionName collection-name)
                  .build)]
    (check-response! (.loadCollection msc param) "loadCollection")
    (log/info "Loaded collection:" collection-name)
    {:collection-name collection-name :status :loaded}))

(defn- do-insert
  [^MilvusServiceClient msc collection-name records opts]
  (let [{:keys [partition]} opts
        columns (rows->columns records)
        fields  (->insert-fields columns)
        builder (doto (InsertParam/newBuilder)
                  (.withCollectionName collection-name)
                  (.withFields fields))]
    (when partition (.withPartitionName builder partition))
    (check-response! (.insert msc (.build builder)) "insert")
    (log/debug "Inserted" (count records) "records into" collection-name)
    {:count (count records) :mutation :insert}))

(defn- do-upsert
  [^MilvusServiceClient msc collection-name records opts]
  (let [{:keys [partition]} opts
        columns (rows->columns records)
        fields  (->insert-fields columns)
        builder (doto (UpsertParam/newBuilder)
                  (.withCollectionName collection-name)
                  (.withFields fields))]
    (when partition (.withPartitionName builder partition))
    (check-response! (.upsert msc (.build builder)) "upsert")
    (log/debug "Upserted" (count records) "records into" collection-name)
    {:count (count records) :mutation :upsert}))

(defn- do-delete
  [^MilvusServiceClient msc collection-name ids opts]
  (let [{:keys [partition]} opts
        id-expr (str "id in ["
                     (str/join ", " (map #(str "\"" % "\"") ids))
                     "]")
        builder (doto (DeleteParam/newBuilder)
                  (.withCollectionName collection-name)
                  (.withExpr id-expr))]
    (when partition (.withPartitionName builder partition))
    (check-response! (.delete msc (.build builder)) "delete")
    (log/debug "Deleted" (count ids) "records from" collection-name)
    {:deleted (count ids)}))

(defn- do-get
  [^MilvusServiceClient msc collection-name ids opts]
  (let [{:keys [include consistency-level] :or {consistency-level :strong}} opts
        id-expr    (str "id in ["
                        (str/join ", " (map #(str "\"" % "\"") ids))
                        "]")
        out-fields (vec (or include
                            ["id" "document" "type" "tags" "content"
                             "content_hash" "created" "updated" "duration"
                             "expires" "access_count" "helpful_count"
                             "unhelpful_count" "project_id"]))
        builder    (doto (QueryParam/newBuilder)
                     (.withCollectionName collection-name)
                     (.withExpr id-expr)
                     (.withOutFields (ArrayList. ^java.util.Collection out-fields)))
        _          (when-let [cl (resolve-consistency consistency-level)]
                     (.withConsistencyLevel builder cl))
        param      (.build builder)
        response   (check-response! (.query msc param) "get")
        wrapper    (QueryResultsWrapper. response)]
    (mapv (fn [row]
            (reduce (fn [m field]
                      (assoc m (keyword field) (.get row field)))
                    {}
                    out-fields))
          (.getRowRecords wrapper))))

(defn- do-query
  [^MilvusServiceClient msc collection-name q-map]
  (let [{:keys [vector limit metric-type output-fields
                filter params partition consistency-level]
         :or   {limit 10 metric-type :cosine}} q-map
        query-vec  (coerce-embedding vector)
        vectors    (ArrayList. ^java.util.Collection [query-vec])
        out-fields (vec (or output-fields
                            ["id" "document" "type" "tags" "content"
                             "content_hash" "created" "updated" "project_id"]))
        builder    (doto (SearchParam/newBuilder)
                     (.withCollectionName collection-name)
                     (.withFloatVectors vectors)
                     (.withVectorFieldName "embedding")
                     (.withTopK (int limit))
                     (.withMetricType (idx/->metric-type metric-type))
                     (.withOutFields (ArrayList. ^java.util.Collection out-fields)))]
    (when-let [cl (resolve-consistency consistency-level)]
      (.withConsistencyLevel builder cl))
    (when filter    (.withExpr builder filter))
    (when params    (.withParams builder (json/write-str params)))
    (when partition (.withPartitionName builder partition))

    (let [response  (check-response! (.search msc (.build builder)) "search")
          wrapper   (SearchResultsWrapper. (.getResults response))
          id-scores (.getIDScore wrapper 0)]
      (if (empty? id-scores)
        []
        (let [field-data (reduce
                          (fn [acc field-name]
                            (try
                              (assoc acc field-name
                                     (vec (.getFieldData wrapper field-name 0)))
                              (catch Exception _e
                                acc)))
                          {}
                          out-fields)
              n (count id-scores)]
          (mapv (fn [i]
                  (let [score (.get id-scores i)
                        base  {:id       (.getStrID score)
                               :distance (.getScore score)}]
                    (reduce (fn [m [field-name values]]
                              (if (< i (count values))
                                (assoc m (keyword field-name) (nth values i))
                                m))
                            base
                            field-data)))
                (range n)))))))

(defn- do-query-scalar
  [^MilvusServiceClient msc collection-name q-map]
  (let [{:keys [filter output-fields limit offset partition consistency-level]
         :or   {limit 100}} q-map
        out-fields (vec (or output-fields
                            ["id" "document" "type" "tags" "content"
                             "content_hash" "created" "updated" "project_id"]))
        builder    (doto (QueryParam/newBuilder)
                     (.withCollectionName collection-name)
                     (.withExpr filter)
                     (.withOutFields (ArrayList. ^java.util.Collection out-fields))
                     (.withLimit (long limit)))]
    (when-let [cl (resolve-consistency consistency-level)]
      (.withConsistencyLevel builder cl))
    (when offset (.withOffset builder (long offset)))
    (when partition (.withPartitionName builder partition))
    (let [response (check-response! (.query msc (.build builder)) "query-scalar")
          wrapper  (QueryResultsWrapper. response)]
      (mapv (fn [row]
              (reduce (fn [m field]
                        (assoc m (keyword field) (.get row field)))
                      {}
                      out-fields))
            (.getRowRecords wrapper)))))

(defn- do-create-collection
  [^MilvusServiceClient msc collection-name opts]
  (let [{:keys [schema index description shards-num]
         :or   {description "" shards-num 1}} opts
        coll-schema (schema/collection-schema schema)
        builder     (doto (CreateCollectionParam/newBuilder)
                      (.withCollectionName collection-name)
                      (.withSchema coll-schema)
                      (.withShardsNum (int shards-num)))]
    (when (seq description)
      (.withDescription builder description))
    (check-response! (.createCollection msc (.build builder)) "createCollection")
    (log/info "Created collection:" collection-name)
    (when index
      (let [idx-param (idx/create-index-param
                       (assoc index :collection-name collection-name))]
        (check-response! (.createIndex msc idx-param) "createIndex")
        (log/debug "Created index on" (:field-name index) "for" collection-name)))
    (let [load-param (-> (LoadCollectionParam/newBuilder)
                         (.withCollectionName collection-name)
                         .build)]
      (check-response! (.loadCollection msc load-param) "loadCollection")
      (log/debug "Loaded collection into memory:" collection-name))
    {:collection-name collection-name :status :created}))

(defn- do-drop-collection
  [^MilvusServiceClient msc collection-name]
  (let [param (-> (DropCollectionParam/newBuilder)
                  (.withCollectionName collection-name)
                  .build)]
    (check-response! (.dropCollection msc param) "dropCollection")
    (log/info "Dropped collection:" collection-name)
    {:collection-name collection-name :status :dropped}))

(defn- do-describe
  [^MilvusServiceClient msc collection-name]
  (let [param    (-> (DescribeCollectionParam/newBuilder)
                     (.withCollectionName collection-name)
                     .build)
        response (check-response! (.describeCollection msc param)
                                  "describeCollection")
        wrapper  (DescCollResponseWrapper. response)]
    {:collection-name (.getCollectionName wrapper)
     :fields          (mapv (fn [ft]
                              {:name      (.getName ft)
                               :data-type (str (.getDataType ft))
                               :primary?  (.isPrimaryKey ft)
                               :auto-id?  (.isAutoID ft)})
                            (.getFields wrapper))}))

;; Extras

(defn- do-list-collections
  [^MilvusServiceClient msc]
  (let [param    (-> (ShowCollectionsParam/newBuilder) .build)
        response (check-response! (.showCollections msc param) "showCollections")]
    (vec (.getCollectionNamesList response))))

(defn- do-release-collection
  [^MilvusServiceClient msc collection-name]
  (let [param (-> (ReleaseCollectionParam/newBuilder)
                  (.withCollectionName collection-name)
                  .build)]
    (check-response! (.releaseCollection msc param) "releaseCollection")
    (log/info "Released collection:" collection-name)))

(defn- do-flush-collection
  [^MilvusServiceClient msc collection-name]
  (let [param (-> (FlushParam/newBuilder)
                  (.withCollectionNames
                   (ArrayList. ^java.util.Collection [collection-name]))
                  .build)]
    (check-response! (.flush msc param) "flush")
    (log/debug "Flushed collection:" collection-name)))

(defn- do-create-index
  [^MilvusServiceClient msc collection-name index-opts]
  (let [idx-param (idx/create-index-param
                   (assoc index-opts :collection-name collection-name))]
    (check-response! (.createIndex msc idx-param) "createIndex")
    (log/info "Created index on" (:field-name index-opts) "for" collection-name)
    {:collection-name collection-name
     :field-name      (:field-name index-opts)
     :status          :indexed}))

(defn- do-drop-index
  [^MilvusServiceClient msc collection-name field-name]
  (let [param (idx/drop-index-param collection-name field-name)]
    (check-response! (.dropIndex msc param) "dropIndex")
    (log/info "Dropped index on" field-name "from" collection-name)))

;; ============================================================================
;; Record + open
;; ============================================================================

(defrecord GrpcClient [^MilvusServiceClient msc opts]
  client/IMilvusCore
  (-has-collection  [_ n]    (do-has-collection msc n))
  (-load-collection [_ n]    (do-load-collection msc n))
  (-insert       [_ n rs o]  (do-insert msc n rs o))
  (-upsert       [_ n rs o]  (do-upsert msc n rs o))
  (-delete       [_ n ids o] (do-delete msc n ids o))
  (-get          [_ n ids o] (do-get msc n ids o))
  (-query        [_ n q]     (do-query msc n q))
  (-query-scalar [_ n q]     (do-query-scalar msc n q))
  (-close [_]
    (when msc
      (try (.close msc) (catch Throwable _ nil))
      (log/info "gRPC client closed")))

  client/IMilvusAdmin
  (-create-collection [_ n o] (do-create-collection msc n o))
  (-drop-collection   [_ n]   (do-drop-collection msc n))
  (-describe          [_ n]   (do-describe msc n))

  client/IMilvusExtras
  (-list-collections   [_]     (do-list-collections msc))
  (-release-collection [_ n]   (do-release-collection msc n))
  (-flush-collection   [_ n]   (do-flush-collection msc n))
  (-create-index       [_ n o] (do-create-index msc n o))
  (-drop-index         [_ n f] (do-drop-index msc n f)))

(defn open
  "Build a `GrpcClient` from the config map.

   Reads agnostic keys (`:host`, `:port`, `:token`, `:database`) from the
   top level and gRPC-specific keys from `(:grpc opts)`. Defaults mirror
   the legacy `milvus-clj.config/default-config` shape for zero behavior
   change when the compat shim calls this."
  [opts]
  (let [{:keys [host port token database]} opts
        grpc (:grpc opts)
        {:keys [connect-timeout-ms keep-alive-time-ms
                keep-alive-timeout-ms keep-alive-without-calls?
                idle-timeout-ms secure]
         :or   {connect-timeout-ms        10000
                keep-alive-time-ms        10000
                keep-alive-timeout-ms     20000
                keep-alive-without-calls? true
                idle-timeout-ms           300000
                secure                    false}} grpc
        builder (doto (ConnectParam/newBuilder)
                  (.withHost (or host "localhost"))
                  (.withPort (int (or port 19530)))
                  (.withConnectTimeout (long connect-timeout-ms) TimeUnit/MILLISECONDS)
                  (.withKeepAliveTime (long keep-alive-time-ms) TimeUnit/MILLISECONDS)
                  (.withKeepAliveTimeout (long keep-alive-timeout-ms) TimeUnit/MILLISECONDS)
                  (.keepAliveWithoutCalls (boolean keep-alive-without-calls?))
                  (.withIdleTimeout (long idle-timeout-ms) TimeUnit/MILLISECONDS)
                  (.withSecure (boolean secure)))]
    (when token    (.withToken builder token))
    (when database (.withDatabaseName builder database))
    (let [msc (MilvusServiceClient. (.build builder))]
      (log/info "gRPC client opened at" (str host ":" port))
      (->GrpcClient msc opts))))

;; ============================================================================
;; Error classification — called by milvus-clj.client/classify-error
;; ============================================================================

(defn classify
  "Classify a gRPC-thrown exception. Returns one of
     :connection-failure  — channel dead, reconnect needed
     :retryable           — transient, retry in place
     :fatal               — schema / auth / caller bug"
  [^Throwable ex]
  (cond
    (instance? StatusRuntimeException ex)
    (let [msg (str (.getMessage ex))]
      (cond
        (or (re-find #"UNAVAILABLE" msg)
            (re-find #"Keepalive failed" msg)
            (re-find #"connection is likely gone" msg)
            (re-find #"io exception" msg))
        :connection-failure

        (or (re-find #"DEADLINE_EXCEEDED" msg)
            (re-find #"RESOURCE_EXHAUSTED" msg))
        :retryable

        :else :fatal))

    (let [msg (some-> ex .getMessage str)]
      (and msg (or (re-find #"not connected" msg)
                   (re-find #"Keepalive failed" msg))))
    :connection-failure

    :else :fatal))
