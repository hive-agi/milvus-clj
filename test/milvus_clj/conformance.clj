(ns milvus-clj.conformance
  "Shared assertions every `IMilvusClient` transport must pass.

   Used by `milvus_clj.transport_conformance_test` to run the same suite
   against both `:grpc` and `:http` via the `*transport*` dynamic var.

   Skipping rule: if no live Milvus is reachable on the transport's port,
   the assertions skip via `milvus-reachable?` rather than fail. That
   keeps CI green when Milvus isn't port-forwarded, while still running
   the suite against both transports in the rare fully-wired case.

   The suite intentionally exercises only ops implemented by BOTH
   transports — `IMilvusCore` + `IMilvusAdmin`. `IMilvusExtras` is
   unimplemented for HTTP (by design; see transport/http.clj)."
  (:require [clojure.test :refer [is testing]]
            [milvus-clj.client :as client]))

(def ^:dynamic *transport* :grpc)

(defn transport-port
  [t]
  (case t :grpc 19530 :http 9091))

(defn make-test-client
  "Build a client for the currently bound `*transport*`.
   Token defaults to `root:Milvus` (matching the dev compose env)."
  ([] (make-test-client {}))
  ([overrides]
   (client/make (merge {:transport *transport*
                        :host      "localhost"
                        :port      (transport-port *transport*)
                        :token     (or (System/getenv "MILVUS_TOKEN")
                                       "root:Milvus")
                        :grpc      {}
                        :http      {}}
                       overrides))))

(defn milvus-reachable?
  "Light probe — can we open a socket to the transport's port?"
  [t]
  (try
    (with-open [sock (java.net.Socket.)]
      (.connect sock
                (java.net.InetSocketAddress. "localhost" (int (transport-port t)))
                (int 500))
      true)
    (catch Throwable _ false)))

(defn close-quietly [c]
  (try (client/-close c) (catch Throwable _ nil)))

(defn test-collection-name
  "Per-run collection name so gRPC + HTTP don't collide."
  [suffix]
  (str "milvus_clj_conformance_" (name *transport*) "_" suffix))

;; ---------------------------------------------------------------------------
;; Shared assertions
;; ---------------------------------------------------------------------------

(defn assert-has-collection-returns-boolean
  "`-has-collection` must return a Clojure boolean for both transports."
  []
  (when (milvus-reachable? *transport*)
    (let [c (make-test-client)]
      (try
        (let [result (client/-has-collection c "__definitely_not_a_collection__")]
          (is (or (true? result) (false? result))
              (str "has-collection must return boolean for " *transport*
                   ", got " (pr-str result))))
        (finally (close-quietly c))))))

(defn assert-describe-shape-matches
  "`-describe` must return a map with `:collection-name` and `:fields`,
   where each field map has `:name :data-type :primary? :auto-id?`."
  [coll-name]
  (when (milvus-reachable? *transport*)
    (let [c (make-test-client)]
      (try
        (when (client/-has-collection c coll-name)
          (let [result (client/-describe c coll-name)]
            (is (map? result))
            (is (contains? result :collection-name))
            (is (vector? (:fields result)))
            (doseq [f (:fields result)]
              (testing (str "field shape for " (:name f))
                (is (contains? f :name))
                (is (contains? f :data-type))
                (is (contains? f :primary?))
                (is (contains? f :auto-id?))))))
        (finally (close-quietly c))))))

(defn assert-protocols-satisfied
  "Smoke check: the client instance satisfies the documented protocols.
   Skipped for gRPC when no server is reachable — its constructor
   eagerly opens the channel and blocks for the connect-timeout."
  []
  (when (or (= *transport* :http) (milvus-reachable? *transport*))
    (let [c (make-test-client)]
      (try
        (is (satisfies? client/IMilvusCore c)
            (str *transport* " must satisfy IMilvusCore"))
        (is (satisfies? client/IMilvusAdmin c)
            (str *transport* " must satisfy IMilvusAdmin"))
        (finally (close-quietly c))))))
