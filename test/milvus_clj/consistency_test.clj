(ns milvus-clj.consistency-test
  "Golden, property, and mutation tests for consistency level support.

   Tests the resolve-consistency helper and the consistency-levels map
   added to support read-after-write guarantees on remote Milvus."
  (:require [clojure.test :refer [deftest testing is]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clojure.test.check.clojure-test :refer [defspec]]
            [hive-test.golden :as golden]
            [hive-test.properties :as props]
            [hive-test.mutation :as mut]
            [milvus-clj.api :as milvus])
  (:import [io.milvus.common.clientenum ConsistencyLevelEnum]))

;; =============================================================================
;; Golden: consistency-levels map shape
;; =============================================================================

(golden/deftest-golden consistency-levels-shape
  "test/golden/milvus-clj/consistency-levels.edn"
  (let [levels @(resolve 'milvus-clj.api/consistency-levels)]
    {:keys     (sort (keys levels))
     :strong   (str (:strong levels))
     :bounded  (str (:bounded levels))
     :session  (str (:session levels))
     :eventually (str (:eventually levels))}))

;; =============================================================================
;; Golden: resolve-consistency outputs
;; =============================================================================

(golden/deftest-golden resolve-consistency-shape
  "test/golden/milvus-clj/resolve-consistency.edn"
  (let [resolve-fn @(resolve 'milvus-clj.api/resolve-consistency)]
    {:strong     (str (resolve-fn :strong))
     :bounded    (str (resolve-fn :bounded))
     :session    (str (resolve-fn :session))
     :eventually (str (resolve-fn :eventually))
     :nil-input  (resolve-fn nil)
     :unknown    (str (resolve-fn :nonexistent))}))

;; =============================================================================
;; Property: resolve-consistency is total (never throws)
;; =============================================================================

(def gen-consistency-keyword
  (gen/one-of [(gen/return :strong)
               (gen/return :bounded)
               (gen/return :session)
               (gen/return :eventually)
               (gen/return nil)
               gen/keyword]))

(props/defprop-total resolve-consistency-total
  (deref (resolve 'milvus-clj.api/resolve-consistency))
  gen-consistency-keyword
  {:num-tests 200})

;; =============================================================================
;; Property: valid keywords always produce a ConsistencyLevelEnum
;; =============================================================================

(def gen-valid-consistency-keyword
  (gen/elements [:strong :bounded :session :eventually]))

(defspec prop-valid-keywords-produce-enum 100
  (prop/for-all [kw gen-valid-consistency-keyword]
    (let [resolve-fn @(resolve 'milvus-clj.api/resolve-consistency)]
      (instance? ConsistencyLevelEnum (resolve-fn kw)))))

;; =============================================================================
;; Property: nil input produces nil (no default injection)
;; =============================================================================

(deftest nil-returns-nil
  (let [resolve-fn @(resolve 'milvus-clj.api/resolve-consistency)]
    (is (nil? (resolve-fn nil)))))

;; =============================================================================
;; Property: idempotent — calling resolve twice on the keyword is same as once
;; (resolve returns an enum, not a keyword, so the second call would use the
;;  fallback. Test that the map is closed over valid keywords.)
;; =============================================================================

(defspec prop-all-valid-keywords-mapped 50
  (prop/for-all [kw gen-valid-consistency-keyword]
    (let [levels @(resolve 'milvus-clj.api/consistency-levels)]
      (contains? levels kw))))

;; =============================================================================
;; Mutation: resolve-consistency returns wrong enum (inverted mapping)
;; =============================================================================

(mut/deftest-mutations resolve-consistency-mutations-caught
  milvus-clj.api/resolve-consistency
  [["always-nil"    (fn [_] nil)]
   ["always-strong" (fn [_] ConsistencyLevelEnum/STRONG)]
   ["inverted"      (fn [kw]
                      (case kw
                        :strong ConsistencyLevelEnum/EVENTUALLY
                        :eventually ConsistencyLevelEnum/STRONG
                        :bounded ConsistencyLevelEnum/SESSION
                        :session ConsistencyLevelEnum/BOUNDED
                        nil))]]
  (fn []
    (let [resolve-fn @(resolve 'milvus-clj.api/resolve-consistency)]
      (is (= ConsistencyLevelEnum/STRONG (resolve-fn :strong)))
      (is (= ConsistencyLevelEnum/BOUNDED (resolve-fn :bounded)))
      (is (= ConsistencyLevelEnum/SESSION (resolve-fn :session)))
      (is (= ConsistencyLevelEnum/EVENTUALLY (resolve-fn :eventually)))
      (is (nil? (resolve-fn nil))))))
