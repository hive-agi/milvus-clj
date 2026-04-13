(ns milvus-clj.transport.http
  "HTTP/REST transport for `milvus-clj.client`. Talks Milvus v2.5.x REST API
   on port 9091 via `java.net.http.HttpClient` + `clojure.data.json`.

   Why this exists:
     - HTTP/1.1 with short-lived connections has zero idle state; the
       gVisor netstack idle-sweep (see `transport/grpc.clj` for the gory
       details) cannot kill a connection that doesn't outlive a request.
     - `java.net.http` is the repo's standard HTTP client (see
       `hive-mcp/src/hive_mcp/embeddings/openrouter.clj` for the canonical
       pattern).

   NOTE on `Connection: close`: java.net.http explicitly rejects the
   `Connection` header as restricted (throws `IllegalArgumentException`).
   The JDK's HTTP/1.1 client manages connection reuse internally and the
   default pool's idle timeout is short; combined with our per-record
   client lifetime (one `with-client` scope), idle-state exposure is
   negligible. If a deployment sees gVisor-class idle kills even on
   HTTP, the mitigation is to open a fresh `HttpClient` per request —
   trivial to add later by moving builder construction into `http-post`.

   Implements `IMilvusCore` + `IMilvusAdmin`. `IMilvusExtras` methods throw
   — hive-milvus doesn't call them and PR-2 scope forbids gold-plating.

   Return shapes are LSP-equivalent to `transport/grpc.clj` — every
   protocol method here produces the same keyword-keyed map shape as the
   gRPC `do-*` helpers, so a conformance suite can swap transports
   transparently.

   Loaded lazily via `requiring-resolve` from `milvus-clj.client/make`."
  (:require [milvus-clj.client :as client]
            [clojure.data.json :as json]
            [clojure.string :as str]
            [taoensso.timbre :as log])
  (:import [java.net URI]
           [java.net.http HttpRequest
            HttpRequest$BodyPublishers HttpResponse HttpResponse$BodyHandlers]
           [java.time Duration]
           [java.io IOException]))

;; ============================================================================
;; HTTP plumbing
;; ============================================================================

(defn- base-url
  [{:keys [host port]}]
  (str "http://" (or host "localhost") ":" (or port 9091)))

(defn- build-request
  ^java.net.http.HttpRequest [^String url ^String body opts]
  (let [{:keys [token request-timeout-ms]} opts
        builder (doto (HttpRequest/newBuilder)
                  (.uri (URI/create url))
                  (.header "Content-Type" "application/json")
                  (.header "Accept" "application/json")
                  ;; Note: `Connection` is a restricted header in
                  ;; java.net.http — setting it throws IAE. See ns docstring.
                  (.POST (HttpRequest$BodyPublishers/ofString body))
                  (.timeout (Duration/ofMillis
                             (long (or request-timeout-ms 30000)))))]
    (when token
      (.header builder "Authorization" (str "Bearer " token)))
    (.build builder)))

(defn- http-post
  "POST a JSON body to `<base>/<path>`. Returns the parsed response map
   (keyword keys). Throws ex-info tagged with `::client/transport :http`
   on non-2xx or IO failure."
  [^java.net.http.HttpClient hc opts path body-map]
  (let [url  (str (base-url opts) path)
        body (json/write-str body-map)
        req  (build-request url body opts)]
    (try
      (let [^HttpResponse resp (.send hc req (HttpResponse$BodyHandlers/ofString))
            status   (.statusCode resp)
            body-str (.body resp)]
        (if (<= 200 status 299)
          (try (json/read-str body-str :key-fn keyword)
               (catch Throwable _parse
                 (throw (ex-info (str "Milvus HTTP " path
                                      " returned unparseable body")
                                 {::client/transport :http
                                  :status status
                                  :path   path
                                  :body   body-str}))))
          (throw (ex-info (str "Milvus HTTP " path " failed: " status)
                          {::client/transport :http
                           :status status
                           :path   path
                           :body   body-str}))))
      (catch IOException ioe
        (throw (ex-info (str "Milvus HTTP " path " IO failure: "
                             (.getMessage ioe))
                        {::client/transport :http
                         :path path
                         :cause :io}
                        ioe))))))

(defn- check-body!
  "Milvus REST wraps *everything* in `{:code 0 :data ...}`. Non-zero code
   is a server-side failure that didn't bubble to an HTTP status."
  [body path]
  (let [code (:code body)]
    (when (and code (not (zero? code)))
      (throw (ex-info (str "Milvus HTTP " path " returned code=" code
                           " message=" (:message body))
                      {::client/transport :http
                       :code    code
                       :message (:message body)
                       :path    path})))
    body))

(defn- post!
  "POST + server-code check. Returns `:data` from the envelope."
  [hc opts path body-map]
  (:data (check-body! (http-post hc opts path body-map) path)))

;; ============================================================================
;; Schema / index rendering — Clojure maps → REST JSON
;; ============================================================================

(def ^:private kw->rest-type
  "Our internal type keyword → Milvus REST v2 DataType string."
  {:bool          "Bool"
   :int8          "Int8"
   :int16         "Int16"
   :int32         "Int32"
   :int64         "Int64"
   :float         "Float"
   :double        "Double"
   :varchar       "VarChar"
   :json          "JSON"
   :array         "Array"
   :float-vector  "FloatVector"
   :binary-vector "BinaryVector"})

(defn- field->rest
  [{:keys [name type primary? auto-id? max-length dimension description]}]
  (cond-> {:fieldName  (clojure.core/name name)
           :dataType   (or (get kw->rest-type type) (str type))
           :isPrimary  (boolean primary?)
           :autoID     (boolean auto-id?)}
    description (assoc :description (str description))
    (or max-length dimension)
    (assoc :elementTypeParams
           (cond-> {}
             max-length (assoc :max_length (int max-length))
             dimension  (assoc :dim (int dimension))))))

(defn- schema->rest
  [fields]
  {:fields (mapv field->rest fields)})

(def ^:private kw->metric
  {:l2 "L2" :ip "IP" :cosine "COSINE"})

(def ^:private kw->index-type
  {:flat      "FLAT"
   :ivf-flat  "IVF_FLAT"
   :ivf-sq8   "IVF_SQ8"
   :ivf-pq    "IVF_PQ"
   :hnsw      "HNSW"
   :diskann   "DISKANN"
   :autoindex "AUTOINDEX"})

(defn- index->rest
  [{:keys [field-name index-name index-type metric-type extra-params]}]
  (cond-> {:fieldName  (clojure.core/name field-name)
           :metricType (or (get kw->metric metric-type) (str metric-type))}
    index-name   (assoc :indexName (clojure.core/name index-name))
    index-type   (assoc :indexType (or (get kw->index-type index-type)
                                       (str index-type)))
    (seq extra-params) (assoc :params extra-params)))

;; ============================================================================
;; Shared helpers
;; ============================================================================

(def ^:private default-output-fields
  ["id" "document" "type" "tags" "content"
   "content_hash" "created" "updated" "project_id"])

(def ^:private default-get-fields
  ["id" "document" "type" "tags" "content"
   "content_hash" "created" "updated" "duration"
   "expires" "access_count" "helpful_count"
   "unhelpful_count" "project_id"])

(defn- with-db
  "Attach :dbName if the client has one configured."
  [body {:keys [database]}]
  (cond-> body database (assoc :dbName database)))

(defn- row->entry
  "Milvus REST returns rows as maps with string keys. Normalize to keyword
   keys to match the gRPC transport's `do-get` / `do-query-scalar` shape."
  [row]
  (reduce-kv (fn [m k v] (assoc m (keyword (name k)) v)) {} row))

;; ============================================================================
;; Operation bodies — mirror transport/grpc.clj do-* return shapes exactly
;; ============================================================================

(defn- do-has-collection
  [hc opts collection-name]
  (let [body (with-db {:collectionName collection-name} opts)
        data (post! hc opts "/v2/vectordb/collections/has" body)]
    (boolean (:has data))))

(defn- do-load-collection
  [hc opts collection-name]
  (let [body (with-db {:collectionName collection-name} opts)]
    (post! hc opts "/v2/vectordb/collections/load" body)
    (log/info "Loaded collection:" collection-name)
    {:collection-name collection-name :status :loaded}))

(defn- do-insert
  [hc opts collection-name records _op-opts]
  (let [body (with-db {:collectionName collection-name
                       :data           (vec records)} opts)]
    (post! hc opts "/v2/vectordb/entities/insert" body)
    (log/debug "Inserted" (count records) "records into" collection-name)
    {:count (count records) :mutation :insert}))

(defn- do-upsert
  [hc opts collection-name records _op-opts]
  (let [body (with-db {:collectionName collection-name
                       :data           (vec records)} opts)]
    (post! hc opts "/v2/vectordb/entities/upsert" body)
    (log/debug "Upserted" (count records) "records into" collection-name)
    {:count (count records) :mutation :upsert}))

(defn- id-in-filter
  [ids]
  (str "id in ["
       (str/join ", " (map #(str "\"" % "\"") ids))
       "]"))

(defn- do-delete
  [hc opts collection-name ids _op-opts]
  (let [body (with-db {:collectionName collection-name
                       :filter         (id-in-filter ids)} opts)]
    (post! hc opts "/v2/vectordb/entities/delete" body)
    (log/debug "Deleted" (count ids) "records from" collection-name)
    {:deleted (count ids)}))

(defn- do-get
  [hc opts collection-name ids op-opts]
  (let [{:keys [include]} op-opts
        out-fields (vec (or include default-get-fields))
        body (with-db {:collectionName collection-name
                       :id             (vec ids)
                       :outputFields   out-fields} opts)
        data (post! hc opts "/v2/vectordb/entities/get" body)]
    (mapv row->entry (or data []))))

(defn- do-query
  [hc opts collection-name q-map]
  (let [{:keys [vector limit metric-type output-fields
                filter params partition]
         :or   {limit 10 metric-type :cosine}} q-map
        out-fields (vec (or output-fields default-output-fields))
        base {:collectionName collection-name
              :data           [(vec vector)]
              :annsField      "embedding"
              :limit           (int limit)
              :outputFields    out-fields
              :metricType      (or (get kw->metric metric-type)
                                   (str metric-type))}
        body (cond-> base
               filter         (assoc :filter filter)
               (seq params)   (assoc :searchParams params)
               partition      (assoc :partitionNames [partition]))
        body (with-db body opts)
        data (post! hc opts "/v2/vectordb/entities/search" body)]
    (mapv (fn [row]
            (let [entry (row->entry row)
                  ;; REST returns :distance for COSINE/L2/IP under either
                  ;; "distance" or "score" depending on version.
                  dist  (or (:distance entry) (:score entry))]
              (-> entry
                  (assoc :distance (when dist (double dist)))
                  (dissoc :score))))
          (or data []))))

(defn- do-query-scalar
  [hc opts collection-name q-map]
  (let [{:keys [filter output-fields limit offset partition]
         :or   {limit 100}} q-map
        out-fields (vec (or output-fields default-output-fields))
        base {:collectionName collection-name
              :filter         filter
              :outputFields   out-fields
              :limit          (int limit)}
        body (cond-> base
               offset    (assoc :offset (int offset))
               partition (assoc :partitionNames [partition]))
        body (with-db body opts)
        data (post! hc opts "/v2/vectordb/entities/query" body)]
    (mapv row->entry (or data []))))

(defn- do-create-collection
  [hc opts collection-name op-opts]
  (let [{:keys [schema index description shards-num]} op-opts
        idx-params (when index
                     [(index->rest
                       (merge {:field-name "embedding"
                               :index-name "embedding_idx"}
                              index))])
        body (cond-> {:collectionName collection-name
                      :schema         (schema->rest schema)}
               (seq description) (assoc :description description)
               shards-num        (assoc :shardsNum (int shards-num))
               (seq idx-params)  (assoc :indexParams idx-params))
        body (with-db body opts)]
    (post! hc opts "/v2/vectordb/collections/create" body)
    (log/info "Created collection:" collection-name)
    ;; REST `/create` loads the collection as part of the call when
    ;; indexParams is provided; call load explicitly otherwise to stay
    ;; behavior-equivalent with gRPC `do-create-collection`.
    (when-not (seq idx-params)
      (post! hc opts "/v2/vectordb/collections/load"
             (with-db {:collectionName collection-name} opts)))
    {:collection-name collection-name :status :created}))

(defn- do-drop-collection
  [hc opts collection-name]
  (let [body (with-db {:collectionName collection-name} opts)]
    (post! hc opts "/v2/vectordb/collections/drop" body)
    (log/info "Dropped collection:" collection-name)
    {:collection-name collection-name :status :dropped}))

(defn- rest-field->map
  [f]
  (let [pk? (or (:primaryKey f) (:isPrimary f) (:primary f))
        auto (or (:autoId f) (:autoID f))
        dtype (or (:type f) (:dataType f))]
    {:name      (or (:name f) (:fieldName f))
     :data-type (str dtype)
     :primary?  (boolean pk?)
     :auto-id?  (boolean auto)}))

(defn- do-describe
  [hc opts collection-name]
  (let [body (with-db {:collectionName collection-name} opts)
        data (post! hc opts "/v2/vectordb/collections/describe" body)
        fields (or (:fields data)
                   (get-in data [:schema :fields])
                   [])]
    {:collection-name (or (:collectionName data) collection-name)
     :fields          (mapv rest-field->map fields)}))

;; ============================================================================
;; Record + open
;; ============================================================================

(defn- extras-unavailable [op]
  (throw (ex-info (str "HTTP transport does not implement " op
                       " (out of scope for PR-2)")
                  {::client/transport :http
                   :op op})))

(defrecord HttpClient [^java.net.http.HttpClient hc opts]
  client/IMilvusCore
  (-has-collection  [_ n]    (do-has-collection hc opts n))
  (-load-collection [_ n]    (do-load-collection hc opts n))
  (-insert       [_ n rs o]  (do-insert hc opts n rs o))
  (-upsert       [_ n rs o]  (do-upsert hc opts n rs o))
  (-delete       [_ n ids o] (do-delete hc opts n ids o))
  (-get          [_ n ids o] (do-get hc opts n ids o))
  (-query        [_ n q]     (do-query hc opts n q))
  (-query-scalar [_ n q]     (do-query-scalar hc opts n q))
  (-close [_]
    ;; java.net.http.HttpClient is GC-managed; no explicit close in JDK 21.
    (log/info "HTTP client closed"))

  client/IMilvusAdmin
  (-create-collection [_ n o] (do-create-collection hc opts n o))
  (-drop-collection   [_ n]   (do-drop-collection hc opts n))
  (-describe          [_ n]   (do-describe hc opts n))

  client/IMilvusExtras
  (-list-collections   [_]     (extras-unavailable :list-collections))
  (-release-collection [_ _]   (extras-unavailable :release-collection))
  (-flush-collection   [_ _]   (extras-unavailable :flush-collection))
  (-create-index       [_ _ _] (extras-unavailable :create-index))
  (-drop-index         [_ _ _] (extras-unavailable :drop-index)))

(defn open
  "Build an `HttpClient` from the config map.

   Reads agnostic keys (`:host`, `:port`, `:token`, `:database`) from the
   top level and HTTP-specific keys (`:request-timeout-ms`,
   `:connect-timeout-ms`) from `(:http opts)`. Default port is 9091."
  [opts]
  (let [http (:http opts)
        {:keys [connect-timeout-ms]
         :or   {connect-timeout-ms 5000}} http
        ;; Default HTTP port is 9091, NOT 19530.
        opts' (cond-> opts
                (nil? (:port opts)) (assoc :port 9091)
                ;; Flatten the relevant http knob into top-level for the
                ;; request builder.
                (:request-timeout-ms http)
                (assoc :request-timeout-ms (:request-timeout-ms http)))
        hc (-> (java.net.http.HttpClient/newBuilder)
               (.connectTimeout (Duration/ofMillis (long connect-timeout-ms)))
               (.build))]
    (log/info "HTTP client opened at" (base-url opts'))
    (->HttpClient hc opts')))

;; ============================================================================
;; Error classification — called by milvus-clj.client/classify-error
;; ============================================================================

(defn classify
  "Classify an HTTP-transport exception. Returns one of
     :connection-failure  — IOException / connect-timeout, reconnect needed
     :retryable           — HTTP 5xx / 429, retry in place
     :fatal               — 4xx, schema/auth/caller bug"
  [^Throwable ex]
  (let [data (ex-data ex)
        status (:status data)
        cause  (:cause data)]
    (cond
      (= :io cause) :connection-failure
      (instance? IOException ex) :connection-failure
      (and status (or (<= 500 status 599) (= 429 status))) :retryable
      (and status (<= 400 status 499)) :fatal
      :else :fatal)))
