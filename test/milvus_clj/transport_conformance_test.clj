(ns milvus-clj.transport-conformance-test
  "Runs the conformance suite against every transport.

   LSP-by-construction: any return-shape divergence between gRPC and HTTP
   trips the same assertion twice and fails the suite. Adding a third
   transport is one entry in the `doseq` below."
  (:require [clojure.test :refer [deftest testing use-fixtures]]
            [milvus-clj.conformance :as conf]))

(defn with-each-transport
  "Run each test body once per supported transport."
  [f]
  (doseq [t [:grpc :http]]
    (binding [conf/*transport* t]
      (testing (str "transport=" t)
        (f)))))

(use-fixtures :each with-each-transport)

(deftest protocols-are-satisfied
  (conf/assert-protocols-satisfied))

(deftest has-collection-returns-boolean
  (conf/assert-has-collection-returns-boolean))

(deftest describe-shape-is-consistent
  ;; Uses a well-known collection if one exists; otherwise no-op.
  (conf/assert-describe-shape-matches "memories"))
