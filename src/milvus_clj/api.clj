(ns milvus-clj.api
  "Main public API for milvus-clj, mirroring clojure-chroma-client.api.

   All operations return futures for async compatibility with existing
   Chroma-based code (which uses @(chroma/query ...) pattern).

   Note: `get` shadows clojure.core/get intentionally — callers use it
   as milvus/get which is unambiguous.

   Usage:
     (require '[milvus-clj.api :as milvus])

     (milvus/connect!)
     @(milvus/create-collection \"memories\"
        {:schema milvus-clj.schema/memory-schema-fields
         :index  milvus-clj.index/default-memory-index})
     @(milvus/add \"memories\"
        [{:id \"1\" :embedding [...] :document \"...\" :type \"note\"}])
     @(milvus/query \"memories\"
        {:vector [...] :limit 10 :filter \"type == 'note'\"})
     (milvus/disconnect!)"
  (:refer-clojure :exclude [get])
  (:require [milvus-clj.config :as config]
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
           [io.milvus.param.index CreateIndexParam]
           [io.milvus.response
            SearchResultsWrapper QueryResultsWrapper
            DescCollResponseWrapper]
           [io.milvus.grpc DataType]
           [java.util ArrayList]
           [java.util.concurrent TimeUnit]))

;; ============================================================================
;; Client lifecycle
;; ============================================================================

(defonce ^:private client-atom (atom nil))

(defn- check-response!
  "Validate a Milvus R<T> response. Returns data on success, throws on error.
   The Milvus Java SDK wraps all responses in R<T> with status codes."
  [^R response operation]
  (if (zero? (.getStatus response))
    (.getData response)
    (throw (ex-info (str "Milvus " operation " failed: " (.getMessage response))
                    {:status    (.getStatus response)
                     :message   (.getMessage response)
                     :exception (.getException response)
                     :operation operation}))))

(defn- require-client!
  "Get the active MilvusServiceClient or throw."
  ^MilvusServiceClient []
  (or @client-atom
      (throw (ex-info "Milvus client not connected. Call (connect!) first." {}))))

(defn connect!
  "Connect to Milvus and store the client.
   Uses config from milvus-clj.config, optionally merging override opts.

   Returns the MilvusServiceClient instance.

   Example:
     (connect!)
     (connect! {:host \"milvus.prod\" :port 19530})"
  ([]
   (connect! {}))
  ([opts]
   (when (seq opts) (config/configure! opts))
   (let [{:keys [host port connect-timeout-ms keep-alive-time-ms
                 keep-alive-timeout-ms idle-timeout-ms
                 secure token database]} (config/get-config)
         builder (doto (ConnectParam/newBuilder)
                   (.withHost host)
                   (.withPort (int port))
                   (.withConnectTimeout (long connect-timeout-ms) TimeUnit/MILLISECONDS)
                   (.withKeepAliveTime (long keep-alive-time-ms) TimeUnit/MILLISECONDS)
                   (.withKeepAliveTimeout (long keep-alive-timeout-ms) TimeUnit/MILLISECONDS)
                   (.withIdleTimeout (long idle-timeout-ms) TimeUnit/MILLISECONDS)
                   (.withSecure (boolean secure)))]
     (when token    (.withToken builder token))
     (when database (.withDatabaseName builder database))
     (let [client (MilvusServiceClient. (.build builder))]
       (reset! client-atom client)
       (log/info "Connected to Milvus at" (str host ":" port))
       client))))

(defn disconnect!
  "Close the Milvus client connection and release resources."
  []
  (when-let [^MilvusServiceClient client @client-atom]
    (.close client)
    (reset! client-atom nil)
    (log/info "Disconnected from Milvus")))

(defn connected?
  "Check if a Milvus client is currently connected."
  []
  (some? @client-atom))

(defmacro with-connection
  "Execute body with a temporary Milvus connection that auto-disconnects.

   Example:
     (with-connection {:host \"localhost\" :port 19530}
       @(list-collections))"
  [opts & body]
  `(try
     (connect! ~opts)
     ~@body
     (finally
       (disconnect!))))

;; ============================================================================
;; Collection management
;; ============================================================================

(defn create-collection
  "Create a Milvus collection with explicit schema. Returns a future.

   Unlike Chroma (schemaless), Milvus requires field definitions upfront.
   Optionally creates an index and loads the collection into memory.

   Options:
     :schema      - vector of field maps (see milvus-clj.schema)
     :index       - index config map (see milvus-clj.index/default-memory-index)
     :description - collection description string
     :shards-num  - number of shards (default 1)

   Example:
     @(create-collection \"memories\"
        {:schema milvus-clj.schema/memory-schema-fields
         :index  milvus-clj.index/default-memory-index})"
  [collection-name {:keys [schema index description shards-num]
                    :or   {description "" shards-num 1}}]
  (future
    (let [client      (require-client!)
          coll-schema (schema/collection-schema schema)
          builder     (doto (CreateCollectionParam/newBuilder)
                        (.withCollectionName collection-name)
                        (.withSchema coll-schema)
                        (.withShardsNum (int shards-num)))
          _           (when (seq description)
                        (.withDescription builder description))
          create-param (.build builder)]
      (check-response! (.createCollection client create-param) "createCollection")
      (log/info "Created collection:" collection-name)

      ;; Create index if specified
      (when index
        (let [idx-param (idx/create-index-param
                          (assoc index :collection-name collection-name))]
          (check-response! (.createIndex client idx-param) "createIndex")
          (log/debug "Created index on" (:field-name index) "for" collection-name)))

      ;; Load collection into memory for search
      (let [load-param (-> (LoadCollectionParam/newBuilder)
                           (.withCollectionName collection-name)
                           .build)]
        (check-response! (.loadCollection client load-param) "loadCollection")
        (log/debug "Loaded collection into memory:" collection-name))

      {:collection-name collection-name :status :created})))

(defn has-collection
  "Check if a collection exists. Returns a future resolving to boolean."
  [collection-name]
  (future
    (let [client (require-client!)
          param  (-> (HasCollectionParam/newBuilder)
                     (.withCollectionName collection-name)
                     .build)]
      (check-response! (.hasCollection client param) "hasCollection"))))

(defn get-collection
  "Describe a collection's schema and properties. Returns a future.
   Analogous to chroma/get-collection."
  [collection-name]
  (future
    (let [client   (require-client!)
          param    (-> (DescribeCollectionParam/newBuilder)
                       (.withCollectionName collection-name)
                       .build)
          response (check-response! (.describeCollection client param)
                                    "describeCollection")
          wrapper  (DescCollResponseWrapper. response)]
      {:collection-name (.getCollectionName wrapper)
       :fields          (mapv (fn [ft]
                                {:name      (.getName ft)
                                 :data-type (str (.getDataType ft))
                                 :primary?  (.isPrimaryKey ft)
                                 :auto-id?  (.isAutoID ft)})
                              (.getFields wrapper))})))

(defn list-collections
  "List all collection names. Returns a future resolving to a vector of strings."
  []
  (future
    (let [client   (require-client!)
          param    (-> (ShowCollectionsParam/newBuilder) .build)
          response (check-response! (.showCollections client param)
                                    "showCollections")]
      (vec (.getCollectionNamesList response)))))

(defn drop-collection
  "Drop (delete) a collection. Returns a future.
   WARNING: This permanently deletes all data in the collection."
  [collection-name]
  (future
    (let [client (require-client!)
          param  (-> (DropCollectionParam/newBuilder)
                     (.withCollectionName collection-name)
                     .build)]
      (check-response! (.dropCollection client param) "dropCollection")
      (log/info "Dropped collection:" collection-name)
      {:collection-name collection-name :status :dropped})))

(defn load-collection
  "Load a collection into memory for search/query. Returns a future.
   Milvus requires collections to be loaded before vector search."
  [collection-name]
  (future
    (let [client (require-client!)
          param  (-> (LoadCollectionParam/newBuilder)
                     (.withCollectionName collection-name)
                     .build)]
      (check-response! (.loadCollection client param) "loadCollection")
      (log/info "Loaded collection:" collection-name))))

(defn release-collection
  "Release a collection from memory. Returns a future.
   Frees memory but collection must be reloaded before search."
  [collection-name]
  (future
    (let [client (require-client!)
          param  (-> (ReleaseCollectionParam/newBuilder)
                     (.withCollectionName collection-name)
                     .build)]
      (check-response! (.releaseCollection client param) "releaseCollection")
      (log/info "Released collection:" collection-name))))

;; ============================================================================
;; Data operations (CRUD)
;; ============================================================================

(defn- rows->columns
  "Convert row-oriented data (vector of maps) to columnar format.
   Milvus SDK v1 requires columnar insertion via InsertParam.Field.

   Input:  [{:id \"1\" :content \"a\"} {:id \"2\" :content \"b\"}]
   Output: {:id [\"1\" \"2\"] :content [\"a\" \"b\"]}"
  [rows]
  (when (seq rows)
    (let [field-names (keys (first rows))]
      (reduce (fn [acc field]
                (assoc acc field (mapv #(clojure.core/get % field) rows)))
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

(defn add
  "Insert or upsert records into a collection. Returns a future.

   Records are row-oriented maps, converted to columnar format internally.
   This mirrors the chroma/add API shape.

   Options:
     :upsert?   - if true, upsert instead of insert (default false)
     :partition  - optional partition name (Milvus-specific)

   Example:
     @(add \"memories\"
        [{:id \"entry-1\" :embedding [0.1 0.2 ...] :document \"hello\" :type \"note\"}
         {:id \"entry-2\" :embedding [0.3 0.4 ...] :document \"world\" :type \"decision\"}]
        :upsert? true)"
  [collection-name records & {:keys [upsert? partition]
                               :or   {upsert? false}}]
  (future
    (let [client  (require-client!)
          columns (rows->columns records)
          fields  (->insert-fields columns)]
      (if upsert?
        (let [builder (doto (UpsertParam/newBuilder)
                        (.withCollectionName collection-name)
                        (.withFields fields))]
          (when partition (.withPartitionName builder partition))
          (let [resp (check-response! (.upsert client (.build builder)) "upsert")]
            (log/debug "Upserted" (count records) "records into" collection-name)
            {:count    (count records)
             :mutation :upsert}))
        (let [builder (doto (InsertParam/newBuilder)
                        (.withCollectionName collection-name)
                        (.withFields fields))]
          (when partition (.withPartitionName builder partition))
          (let [resp (check-response! (.insert client (.build builder)) "insert")]
            (log/debug "Inserted" (count records) "records into" collection-name)
            {:count    (count records)
             :mutation :insert}))))))

(defn get
  "Get records by IDs from a collection. Returns a future.

   Uses scalar query with ID filter under the hood (Milvus pattern).
   Analogous to chroma/get with :ids.

   Options:
     :include - set/vector of field names to return (default: all scalar fields)

   Example:
     @(get \"memories\" [\"entry-1\" \"entry-2\"] :include [\"content\" \"type\"])"
  [collection-name ids & {:keys [include]}]
  (future
    (let [client     (require-client!)
          id-expr    (str "id in ["
                          (str/join ", " (map #(str "\"" % "\"") ids))
                          "]")
          out-fields (vec (or include
                              ["id" "document" "type" "tags" "content"
                               "content_hash" "created" "updated" "duration"
                               "expires" "access_count" "helpful_count"
                               "unhelpful_count" "project_id"]))
          param      (-> (QueryParam/newBuilder)
                         (.withCollectionName collection-name)
                         (.withExpr id-expr)
                         (.withOutFields (ArrayList. ^java.util.Collection out-fields))
                         .build)
          response   (check-response! (.query client param) "get")
          wrapper    (QueryResultsWrapper. response)]
      (mapv (fn [row]
              (reduce (fn [m field]
                        (assoc m (keyword field) (.get row field)))
                      {}
                      out-fields))
            (.getRowRecords wrapper)))))

(defn delete
  "Delete records by IDs from a collection. Returns a future.

   Analogous to chroma/delete with :ids.

   Options:
     :partition - optional partition name

   Example:
     @(delete \"memories\" [\"entry-1\" \"entry-2\"])"
  [collection-name ids & {:keys [partition]}]
  (future
    (let [client  (require-client!)
          id-expr (str "id in ["
                       (str/join ", " (map #(str "\"" % "\"") ids))
                       "]")
          builder (doto (DeleteParam/newBuilder)
                    (.withCollectionName collection-name)
                    (.withExpr id-expr))]
      (when partition (.withPartitionName builder partition))
      (let [_resp (check-response! (.delete client (.build builder)) "delete")]
        (log/debug "Deleted" (count ids) "records from" collection-name)
        {:deleted (count ids)}))))

;; ============================================================================
;; Search / Query
;; ============================================================================

(defn query
  "Vector similarity search. Returns a future.

   This is the primary search operation, analogous to chroma/query.
   Combines vector similarity with optional scalar filtering.

   Required in opts:
     :vector       - query vector (seq of numbers)

   Optional in opts:
     :limit         - max results (default 10)
     :metric-type   - distance metric keyword (default :cosine)
     :output-fields - fields to return (default: common scalar fields)
     :filter        - boolean filter expression in Milvus syntax
                      e.g. \"type == 'note'\"
                      e.g. \"type in ['note', 'decision'] and project_id == 'hive-mcp'\"
     :params        - search params map (e.g. {:ef 64} for HNSW)
     :partition     - partition name

   Example:
     @(query \"memories\"
        {:vector [0.1 0.2 ...]
         :limit 10
         :filter \"type == 'note'\"
         :output-fields [\"id\" \"content\" \"type\"]})"
  [collection-name {:keys [vector limit metric-type output-fields
                           filter params partition]
                    :or   {limit 10 metric-type :cosine}}]
  (future
    (let [client    (require-client!)
          query-vec (coerce-embedding vector)
          vectors   (ArrayList. ^java.util.Collection [query-vec])
          out-fields (vec (or output-fields
                              ["id" "document" "type" "tags" "content"
                               "content_hash" "created" "updated" "project_id"]))
          builder   (doto (SearchParam/newBuilder)
                      (.withCollectionName collection-name)
                      (.withFloatVectors vectors)
                      (.withVectorFieldName "embedding")
                      (.withTopK (int limit))
                      (.withMetricType (idx/->metric-type metric-type))
                      (.withOutFields (ArrayList. ^java.util.Collection out-fields)))]
      (when filter    (.withExpr builder filter))
      (when params    (.withParams builder (json/write-str params)))
      (when partition (.withPartitionName builder partition))

      (let [response  (check-response! (.search client (.build builder)) "search")
            wrapper   (SearchResultsWrapper. (.getResults response))
            id-scores (.getIDScore wrapper 0)]
        (if (empty? id-scores)
          []
          ;; Fetch field data for each output field
          (let [field-data (reduce
                             (fn [acc field-name]
                               (try
                                 (assoc acc field-name
                                        (vec (.getFieldData wrapper field-name 0)))
                                 (catch Exception _e
                                   ;; Field might not be in results
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
                  (range n))))))))

(defn query-scalar
  "Scalar (non-vector) query with filter expression. Returns a future.

   This is for metadata-only queries without vector similarity.
   Analogous to chroma/get with :where clauses.

   Milvus filter syntax (differs from Chroma where-clause):
     \"type == 'decision'\"
     \"type in ['note', 'decision']\"
     \"access_count > 5 and project_id == 'hive-mcp'\"
     \"tags like '%migration%'\"

   Options:
     :filter        - boolean expression (required)
     :output-fields - fields to return
     :limit         - max results (default 100)
     :partition     - partition name

   Example:
     @(query-scalar \"memories\"
        {:filter \"type == 'decision' and project_id == 'hive-mcp'\"
         :output-fields [\"id\" \"content\" \"type\"]
         :limit 100})"
  [collection-name {:keys [filter output-fields limit partition]
                    :or   {limit 100}}]
  (future
    (let [client     (require-client!)
          out-fields (vec (or output-fields
                              ["id" "document" "type" "tags" "content"
                               "content_hash" "created" "updated" "project_id"]))
          builder    (doto (QueryParam/newBuilder)
                       (.withCollectionName collection-name)
                       (.withExpr filter)
                       (.withOutFields (ArrayList. ^java.util.Collection out-fields))
                       (.withLimit (long limit)))]
      (when partition (.withPartitionName builder partition))
      (let [response (check-response! (.query client (.build builder)) "query-scalar")
            wrapper  (QueryResultsWrapper. response)]
        (mapv (fn [row]
                (reduce (fn [m field]
                          (assoc m (keyword field) (.get row field)))
                        {}
                        out-fields))
              (.getRowRecords wrapper))))))

;; ============================================================================
;; Convenience operations
;; ============================================================================

(defn flush-collection
  "Flush a collection to persist data to storage. Returns a future.
   In Milvus 2.2+, data is auto-flushed, but explicit flush ensures durability."
  [collection-name]
  (future
    (let [client (require-client!)
          param  (-> (FlushParam/newBuilder)
                     (.withCollectionNames (ArrayList. ^java.util.Collection [collection-name]))
                     .build)]
      (check-response! (.flush client param) "flush")
      (log/debug "Flushed collection:" collection-name))))

(defn collection-stats
  "Get basic statistics about a collection. Returns a future."
  [collection-name]
  (future
    (try
      {:exists? @(has-collection collection-name)
       :info    @(get-collection collection-name)}
      (catch Exception e
        {:error (str e)}))))

(defn create-index
  "Create an index on a collection field. Returns a future.
   Typically called after collection creation if not done inline.

   Example:
     @(create-index \"memories\" milvus-clj.index/default-memory-index)"
  [collection-name index-opts]
  (future
    (let [client    (require-client!)
          idx-param (idx/create-index-param
                      (assoc index-opts :collection-name collection-name))]
      (check-response! (.createIndex client idx-param) "createIndex")
      (log/info "Created index on" (:field-name index-opts) "for" collection-name)
      {:collection-name collection-name
       :field-name      (:field-name index-opts)
       :status          :indexed})))

(defn drop-index
  "Drop an index from a collection field. Returns a future."
  [collection-name field-name]
  (future
    (let [client (require-client!)
          param  (idx/drop-index-param collection-name field-name)]
      (check-response! (.dropIndex client param) "dropIndex")
      (log/info "Dropped index on" field-name "from" collection-name))))
