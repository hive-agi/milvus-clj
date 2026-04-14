(ns milvus-clj.consistency-trifecta-test
  "Same coverage as consistency-test but via deftrifecta — 3 lines, not 60.

   Demonstrates the trifecta macro: one declaration generates golden,
   property, and mutation tests for resolve-consistency."
  (:require [clojure.test :refer [deftest is]]
            [clojure.test.check.generators :as gen]
            [hive-test.trifecta :refer [deftrifecta]]
            [milvus-clj.api :as milvus])
  (:import [io.milvus.common.clientenum ConsistencyLevelEnum]))

;; One declaration → three tests:
;;   resolve-consistency-golden      (4 case assertions + snapshot)
;;   resolve-consistency-property    (200 random inputs, never throws)
;;   resolve-consistency-mutations   (3 mutants × 4 cases = 12 kill checks)

(deftrifecta resolve-consistency
  milvus-clj.api/resolve-consistency
  {:golden-path "test/golden/milvus-clj/trifecta-resolve-consistency.edn"
   :cases       {:strong     :strong
                 :bounded    :bounded
                 :session    :session
                 :eventually :eventually}
   :xf          str
   :gen         (gen/one-of [(gen/return :strong)
                             (gen/return :bounded)
                             (gen/return :session)
                             (gen/return :eventually)
                             (gen/return nil)
                             gen/keyword])
   :pred        #(or (nil? %) (instance? ConsistencyLevelEnum %))
   :num-tests   200
   :mutations   [["always-nil"    (fn [_] nil)]
                 ["always-strong" (fn [_] ConsistencyLevelEnum/STRONG)]
                 ["inverted"      (fn [kw]
                                    (case kw
                                      :strong ConsistencyLevelEnum/EVENTUALLY
                                      :eventually ConsistencyLevelEnum/STRONG
                                      :bounded ConsistencyLevelEnum/SESSION
                                      :session ConsistencyLevelEnum/BOUNDED
                                      nil))]]})
