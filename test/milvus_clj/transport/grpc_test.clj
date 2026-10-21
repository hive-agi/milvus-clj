(ns milvus-clj.transport.grpc-test
  "gRPC-transport edge cases.

   PR-2 scope: minimal smoke tests only — factory dispatch, protocol
   satisfaction, error-classifier dispatch. The channel recycler and
   `:pool :shared` behavior land in PR-3; this ns has placeholders so
   the file exists at the path the PR plan promises."
  (:require [clojure.test :refer [deftest is]]
            [milvus-clj.client :as client]
            [milvus-clj.transport.grpc :as grpc]
            [milvus-clj.conformance :as conf]))

(defn- grpc-reachable? []
  (try
    (with-open [s (java.net.Socket.)]
      (.connect s (java.net.InetSocketAddress. "127.0.0.1" 19530) 500)
      true)
    (catch Throwable _ false)))

(deftest grpc-client-type-implements-the-protocols
  ;; Offline half of the factory check: the type `make` builds for :grpc
  ;; implements every protocol, without opening a channel.
  (let [cls (Class/forName "milvus_clj.transport.grpc.GrpcClient")]
    (is (extends? client/IMilvusCore cls))
    (is (extends? client/IMilvusAdmin cls))
    (is (extends? client/IMilvusExtras cls))))

(deftest factory-dispatches-to-grpc
  ;; MilvusServiceClient's constructor eagerly opens the channel and
  ;; blocks for ~10 s if unreachable. Live only: MILVUS_LIVE_TESTS set and
  ;; a server listening.
  (if-not (and (conf/live-tests-enabled?) (grpc-reachable?))
    (conf/skip! "grpc factory dispatch")
    (let [c (client/make {:transport :grpc
                          :host "127.0.0.1"
                          :port 19530
                          :grpc {}})]
      (try
        (is (satisfies? client/IMilvusCore c))
        (is (satisfies? client/IMilvusAdmin c))
        (is (satisfies? client/IMilvusExtras c))
        (is (= "milvus_clj.transport.grpc.GrpcClient"
               (.getName (class c))))
        (finally
          (try (client/-close c) (catch Throwable _ nil)))))))

(deftest classify-unavailable-is-connection-failure
  (let [ex (ex-info "UNAVAILABLE: Keepalive failed"
                    {::client/transport :grpc})]
    (is (= :connection-failure (grpc/classify ex)))))

(deftest classify-unknown-is-fatal
  (let [ex (ex-info "something random" {::client/transport :grpc})]
    (is (= :fatal (grpc/classify ex)))))

;; PR-3 placeholders (not implemented):
;;   - recycler-rebuilds-channel
;;   - pool-shared-reuses-client
