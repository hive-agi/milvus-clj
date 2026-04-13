(ns milvus-clj.transport.http-test
  "HTTP-transport edge cases that can run WITHOUT a live Milvus.

   Live-dependent assertions live in the parameterized conformance suite
   (`milvus_clj.transport_conformance_test`). This ns focuses on pure
   behavior: JSON shape rendering, auth-header gating, error classifier
   dispatch, filter DSL round-trip."
  (:require [clojure.test :refer [deftest testing is]]
            [clojure.data.json :as json]
            [milvus-clj.client :as client]
            [milvus-clj.transport.http :as http]))

;; ---------------------------------------------------------------------------
;; open / factory
;; ---------------------------------------------------------------------------

(deftest open-defaults-port-to-9091
  (let [c (http/open {:host "127.0.0.1" :http {}})]
    (try
      (is (= 9091 (:port (:opts c)))
          "HTTP default port must be 9091, not 19530")
      (finally (client/-close c)))))

(deftest open-preserves-explicit-port
  (let [c (http/open {:host "127.0.0.1" :port 18080 :http {}})]
    (try
      (is (= 18080 (:port (:opts c))))
      (finally (client/-close c)))))

(deftest factory-dispatches-to-http
  (let [c (client/make {:transport :http
                        :host "127.0.0.1"
                        :http {}})]
    (try
      (is (satisfies? client/IMilvusCore c))
      (is (satisfies? client/IMilvusAdmin c))
      (is (= "milvus_clj.transport.http.HttpClient"
             (.getName (class c))))
      (finally (client/-close c)))))

;; ---------------------------------------------------------------------------
;; Filter DSL round-trip — the spicy escaping case
;; ---------------------------------------------------------------------------

(deftest filter-dsl-survives-json-encoding
  (testing "Milvus filter strings round-trip through data.json without mangling"
    (let [filter-str "type == \"note\" and tags like \"%foo%\""
          body-map   {:collectionName "memories"
                      :filter         filter-str
                      :limit          10}
          encoded    (json/write-str body-map)
          decoded    (json/read-str encoded :key-fn keyword)]
      (is (= filter-str (:filter decoded))
          "filter string must come back byte-equivalent"))))

;; ---------------------------------------------------------------------------
;; Error classifier
;; ---------------------------------------------------------------------------

(deftest classify-5xx-is-retryable
  (let [ex (ex-info "boom" {::client/transport :http :status 503})]
    (is (= :retryable (http/classify ex)))
    (is (= :retryable (client/classify-error ex)))))

(deftest classify-429-is-retryable
  (let [ex (ex-info "rate" {::client/transport :http :status 429})]
    (is (= :retryable (http/classify ex)))))

(deftest classify-4xx-is-fatal
  (let [ex (ex-info "bad" {::client/transport :http :status 400})]
    (is (= :fatal (http/classify ex))))
  (let [ex (ex-info "auth" {::client/transport :http :status 401})]
    (is (= :fatal (http/classify ex)))))

(deftest classify-io-is-connection-failure
  (let [ex (ex-info "io" {::client/transport :http :cause :io}
                    (java.io.IOException. "connect refused"))]
    (is (= :connection-failure (http/classify ex)))
    (is (= :connection-failure (client/classify-error ex)))))

;; ---------------------------------------------------------------------------
;; ex-data tagging
;; ---------------------------------------------------------------------------

(deftest connection-refused-is-tagged
  ;; Use an almost-certainly-closed port so the socket connect fails fast.
  (let [c (http/open {:host "127.0.0.1" :port 1 :http {:request-timeout-ms 500
                                                       :connect-timeout-ms 500}})]
    (try
      (try
        (client/-has-collection c "whatever")
        (is false "should have thrown on connection-refused")
        (catch clojure.lang.ExceptionInfo ex
          (is (= :http (::client/transport (ex-data ex)))
              "thrown exceptions must carry ::client/transport :http")
          (is (= :connection-failure (client/classify-error ex)))))
      (finally (client/-close c)))))
