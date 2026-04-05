(ns milvus-clj.index
  "Milvus index creation and management.

   Milvus requires explicit index creation on vector fields before search.
   Supports IVF_FLAT, HNSW, DISKANN, and other index types with
   configurable distance metrics.

   Key difference from Chroma: Chroma auto-indexes. Milvus needs explicit
   index creation with type and metric selection."
  (:require [clojure.data.json :as json])
  (:import [io.milvus.param IndexType MetricType]
           [io.milvus.param.index CreateIndexParam DescribeIndexParam DropIndexParam]))

;; ---------------------------------------------------------------------------
;; Enum mappings
;; ---------------------------------------------------------------------------

(def index-types
  "Mapping of keywords to Milvus IndexType enum values.

   Common choices:
     :hnsw     — Best recall, higher memory. Good for <10M vectors.
     :ivf-flat — Good balance. Works with partitions.
     :diskann  — Disk-based. Best for >100M vectors.
     :autoindex — Let Milvus choose (cloud/managed)."
  {:flat      IndexType/FLAT
   :ivf-flat  IndexType/IVF_FLAT
   :ivf-sq8   IndexType/IVF_SQ8
   :ivf-pq    IndexType/IVF_PQ
   :hnsw      IndexType/HNSW
   :diskann   IndexType/DISKANN
   :autoindex IndexType/AUTOINDEX})

(def metric-types
  "Mapping of keywords to Milvus MetricType enum values.

   :cosine — Cosine similarity (normalized). Default for embeddings.
   :l2     — Euclidean distance. Lower = more similar.
   :ip     — Inner product. Higher = more similar."
  {:l2     MetricType/L2
   :ip     MetricType/IP
   :cosine MetricType/COSINE})

;; ---------------------------------------------------------------------------
;; Resolvers
;; ---------------------------------------------------------------------------

(defn ->index-type
  "Resolve keyword or IndexType to IndexType enum."
  [it]
  (cond
    (instance? IndexType it) it
    (keyword? it)            (or (get index-types it)
                                 (throw (ex-info (str "Unknown index type: " it)
                                                 {:index-type it
                                                  :valid      (keys index-types)})))
    :else (throw (ex-info "index-type must be keyword or IndexType"
                          {:got (type it)}))))

(defn ->metric-type
  "Resolve keyword or MetricType to MetricType enum."
  [mt]
  (cond
    (instance? MetricType mt) mt
    (keyword? mt)             (or (get metric-types mt)
                                  (throw (ex-info (str "Unknown metric type: " mt)
                                                  {:metric-type mt
                                                   :valid       (keys metric-types)})))
    :else (throw (ex-info "metric-type must be keyword or MetricType"
                          {:got (type mt)}))))

;; ---------------------------------------------------------------------------
;; Param builders
;; ---------------------------------------------------------------------------

(defn create-index-param
  "Build a CreateIndexParam from a Clojure map.

   Required keys:
     :collection-name - target collection
     :field-name      - field to index (usually the vector field)
     :index-type      - keyword from `index-types` (e.g. :hnsw, :ivf-flat)
     :metric-type     - keyword from `metric-types` (e.g. :cosine, :l2, :ip)

   Optional keys:
     :index-name   - custom index name
     :extra-params - map of index-specific params
                     IVF:  {:nlist 1024}
                     HNSW: {:M 16 :efConstruction 256}

   Example:
     (create-index-param
       {:collection-name \"memories\"
        :field-name      \"embedding\"
        :index-type      :hnsw
        :metric-type     :cosine
        :extra-params    {:M 16 :efConstruction 256}})"
  [{:keys [collection-name field-name index-type metric-type
           index-name extra-params]}]
  (let [builder (doto (CreateIndexParam/newBuilder)
                  (.withCollectionName collection-name)
                  (.withFieldName field-name)
                  (.withIndexType (->index-type index-type))
                  (.withMetricType (->metric-type metric-type)))]
    (when index-name
      (.withIndexName builder index-name))
    (when extra-params
      (.withExtraParam builder (json/write-str extra-params)))
    (.build builder)))

(defn describe-index-param
  "Build a DescribeIndexParam for inspecting an existing index."
  [collection-name field-name]
  (-> (DescribeIndexParam/newBuilder)
      (.withCollectionName collection-name)
      (.withFieldName field-name)
      .build))

(defn drop-index-param
  "Build a DropIndexParam for removing an index."
  [collection-name field-name]
  (-> (DropIndexParam/newBuilder)
      (.withCollectionName collection-name)
      (.withFieldName field-name)
      .build))

;; ---------------------------------------------------------------------------
;; Presets
;; ---------------------------------------------------------------------------

(def default-memory-index
  "Default index configuration for the memory embedding field.
   HNSW with cosine similarity — good balance of speed and recall.

   HNSW params:
     M=16             — connections per node (higher = better recall, more memory)
     efConstruction=256 — build-time search width (higher = better recall, slower build)"
  {:field-name   "embedding"
   :index-type   :hnsw
   :metric-type  :cosine
   :extra-params {:M 16 :efConstruction 256}})
