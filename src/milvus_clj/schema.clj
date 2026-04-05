(ns milvus-clj.schema
  "Milvus schema building: FieldType and CollectionSchema construction.

   Converts idiomatic Clojure maps to Milvus Java SDK builder objects.
   Unlike Chroma (schemaless), Milvus requires explicit field definitions
   with types, dimensions, and constraints before data can be inserted.

   Usage:
     (collection-schema
       [{:name \"id\" :type :varchar :primary? true :max-length 256}
        {:name \"embedding\" :type :float-vector :dimension 1536}
        {:name \"content\" :type :varchar :max-length 65535}]
       :description \"Memory collection\")"
  (:import [io.milvus.grpc DataType]
           [io.milvus.param.collection CollectionSchemaParam FieldType]))

;; ---------------------------------------------------------------------------
;; DataType mapping
;; ---------------------------------------------------------------------------

(def data-types
  "Mapping of keywords to Milvus DataType enum values."
  {:bool          DataType/Bool
   :int8          DataType/Int8
   :int16         DataType/Int16
   :int32         DataType/Int32
   :int64         DataType/Int64
   :float         DataType/Float
   :double        DataType/Double
   :varchar       DataType/VarChar
   :json          DataType/JSON
   :array         DataType/Array
   :float-vector  DataType/FloatVector
   :binary-vector DataType/BinaryVector})

(defn ->data-type
  "Resolve a keyword or DataType instance to a DataType enum value.
   Throws ex-info with valid keys on unknown type."
  [dt]
  (cond
    (instance? DataType dt) dt
    (keyword? dt) (or (get data-types dt)
                      (throw (ex-info (str "Unknown data type: " dt)
                                      {:data-type   dt
                                       :valid-types (keys data-types)})))
    :else (throw (ex-info "data-type must be a keyword or DataType instance"
                          {:got (type dt)}))))

;; ---------------------------------------------------------------------------
;; FieldType builder
;; ---------------------------------------------------------------------------

(defn field-type
  "Build a Milvus FieldType from a Clojure map.

   Required keys:
     :name       - field name (string or keyword)
     :type       - data type keyword (see `data-types`)

   Optional keys:
     :primary?    - is this the primary key? (default false)
     :auto-id?    - auto-generate IDs? (default false)
     :max-length  - max length for VarChar fields
     :dimension   - vector dimension for FloatVector/BinaryVector
     :description - field description string

   Example:
     (field-type {:name \"embedding\" :type :float-vector :dimension 1536})"
  [{:keys [name type primary? auto-id? max-length dimension description]
    :or   {primary? false auto-id? false}}]
  (let [builder (doto (FieldType/newBuilder)
                  (.withName (clojure.core/name name))
                  (.withDataType (->data-type type))
                  (.withPrimaryKey (boolean primary?))
                  (.withAutoID (boolean auto-id?)))]
    (when max-length  (.withMaxLength builder (int max-length)))
    (when dimension   (.withDimension builder (int dimension)))
    (when description (.withDescription builder (str description)))
    (.build builder)))

;; ---------------------------------------------------------------------------
;; CollectionSchema builder
;; ---------------------------------------------------------------------------

(defn collection-schema
  "Build a CollectionSchemaParam from a sequence of field definition maps.

   Each field map is passed to `field-type`.
   Note: description is set on CreateCollectionParam, not on the schema.

   Example:
     (collection-schema
       [{:name \"id\"        :type :varchar      :primary? true :max-length 256}
        {:name \"embedding\" :type :float-vector :dimension 1536}
        {:name \"content\"   :type :varchar      :max-length 65535}])"
  [fields]
  (let [builder (CollectionSchemaParam/newBuilder)]
    (doseq [f fields]
      (.addFieldType builder (field-type f)))
    (.build builder)))

;; ---------------------------------------------------------------------------
;; Preset: hive-mcp memory schema
;; ---------------------------------------------------------------------------

(def memory-schema-fields
  "Default schema fields for hive-mcp memory collections.
   Mirrors the metadata stored in Chroma, translated to Milvus field types.

   Use with `collection-schema`:
     (collection-schema memory-schema-fields)"
  [{:name "id"              :type :varchar      :primary? true :max-length 256}
   {:name "embedding"       :type :float-vector :dimension 1536}
   {:name "document"        :type :varchar      :max-length 65535}
   {:name "type"            :type :varchar      :max-length 64}
   {:name "tags"            :type :varchar      :max-length 4096}
   {:name "content"         :type :varchar      :max-length 65535}
   {:name "content_hash"    :type :varchar      :max-length 128}
   {:name "created"         :type :varchar      :max-length 128}
   {:name "updated"         :type :varchar      :max-length 128}
   {:name "duration"        :type :varchar      :max-length 32}
   {:name "expires"         :type :varchar      :max-length 128}
   {:name "access_count"    :type :int64}
   {:name "helpful_count"   :type :int64}
   {:name "unhelpful_count" :type :int64}
   {:name "project_id"      :type :varchar      :max-length 256}])

(defn with-dimension
  "Return memory-schema-fields with the embedding dimension adjusted.

   Example:
     (collection-schema (with-dimension 768))"
  [dim]
  (mapv (fn [f]
          (if (= (:name f) "embedding")
            (assoc f :dimension dim)
            f))
        memory-schema-fields))
