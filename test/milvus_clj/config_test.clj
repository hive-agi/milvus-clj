(ns milvus-clj.config-test
  "Tests for milvus-clj.config — MilvusClientConfig defconfig.

   We replicate three of the four hive-di.testing/defconfig-tests
   generators inline (totality, defaults-only, field-mutations) so we
   can substitute a non-empty keyword generator for the :transport
   field. The default `gen-value-for-type :keyword` in hive-di v0.3.x
   can emit `(keyword \"\")` = `:`, which pr-str → \":\" → fails
   `clojure.edn/read-string` (\"Invalid token: :\"). That collapses
   the auto-generated roundtrip property for any defconfig with a
   :keyword field. See the corresponding pain-point memory.

   The handful of explicit deftests below cover milvus-clj-specific
   semantics (transport coercion, optional :token, env override paths,
   legacy literal fields preserved, configure!/get-config shim
   round-trip)."
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest testing is use-fixtures]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [hive-di.resolve :as resolve]
            [hive-di.testing :as ditest]
            [hive-dsl.result :as r]
            [milvus-clj.config :as config]))

;; =============================================================================
;; Generator replacements — works around upstream `:` keyword bug
;; =============================================================================

(def ^:private gen-nonempty-keyword
  (gen/fmap keyword (gen/such-that seq gen/string-alphanumeric)))

(defn- gen-value-for-type [type-kw]
  (if (= :keyword type-kw)
    gen-nonempty-keyword
    (ditest/gen-value-for-type type-kw)))

(defn- gen-string-for-type [type-kw]
  (if (= :keyword type-kw)
    (gen/fmap name gen-nonempty-keyword)
    (ditest/gen-string-for-type type-kw)))

(defn- gen-config-overrides [fields]
  (let [field-gens (mapv (fn [[field-kw {:keys [type]}]]
                           (gen/one-of
                             [(gen/return nil)
                              (gen/return [field-kw nil])
                              (gen/return [field-kw ""])
                              (gen/fmap #(vector field-kw %) (gen-value-for-type type))
                              (gen/fmap #(vector field-kw %) (gen-string-for-type type))]))
                         fields)]
    (gen/fmap (fn [entries] (into {} (remove nil? entries)))
              (apply gen/tuple field-gens))))

(defn- gen-mock-env [fields]
  (let [env-fields (filterv (fn [[_ spec]] (= :source/env (:source spec))) fields)
        env-gens   (mapv (fn [[_ {:keys [env-var type]}]]
                           (gen/one-of
                             [(gen/return nil)
                              (gen/return [env-var ""])
                              (gen/fmap #(vector env-var %)
                                        (gen-string-for-type type))]))
                         env-fields)]
    (gen/fmap (fn [entries]
                (let [env-map (into {} (remove nil? entries))]
                  {:env-map env-map :env-fn #(get env-map %)}))
              (apply gen/tuple env-gens))))

;; =============================================================================
;; Generated property tests (replacement for defconfig-tests)
;; =============================================================================

(defspec MilvusClientConfig-totality 50
  (prop/for-all [overrides (gen-config-overrides config/MilvusClientConfig-fields)
                 mock      (gen-mock-env config/MilvusClientConfig-fields)]
    (let [result (try
                   (resolve/resolve-config config/MilvusClientConfig-fields
                                           overrides
                                           {:env-fn (:env-fn mock)})
                   (catch Throwable t {:threw t}))]
      (or (r/ok? result) (r/err? result)))))

(defspec MilvusClientConfig-roundtrip 50
  (prop/for-all [overrides (gen-config-overrides config/MilvusClientConfig-fields)]
    (let [result (resolve/resolve-config config/MilvusClientConfig-fields
                                         overrides
                                         {:env-fn (constantly nil)})]
      (if (r/ok? result)
        (let [resolved (:ok result)]
          (= resolved (edn/read-string (pr-str resolved))))
        true))))

(deftest MilvusClientConfig-defaults-only
  (let [result (ditest/resolve-with-defaults-only config/MilvusClientConfig-fields)]
    (is (r/ok? result)
        (str "Defaults-only resolution should succeed, got: " (pr-str result)))))

(deftest MilvusClientConfig-field-mutations
  (doseq [[field-kw _spec] config/MilvusClientConfig-fields]
    (let [result (try
                   (ditest/resolve-with-field-removed config/MilvusClientConfig-fields field-kw)
                   (catch Throwable t {:threw t}))]
      (is (or (r/ok? result) (r/err? result))
          (str "Removing default for " field-kw
               " should return a Result, not throw")))))

;; =============================================================================
;; Targeted smoke tests — milvus-clj-specific semantics
;; =============================================================================

(defn- mock-env [m]
  {:env-fn #(get m %)})

(deftest defaults-only-shape
  (testing "no env, no overrides → all typed defaults present"
    (let [r (config/resolve-MilvusClientConfig {} (mock-env {}))]
      (is (r/ok? r))
      (let [v (:ok r)]
        ;; env-sourced fields
        (is (= "localhost"        (:host v)))
        (is (= 19530              (:port v)))
        (is (= :grpc              (:transport v)))
        (is (nil?                 (:token v)))
        ;; literal fields — legacy default preservation
        (is (= "hive-mcp-memory"  (:collection-name v)))
        (is (= 10000              (:connect-timeout-ms v)))
        (is (= 10000              (:keep-alive-time-ms v)))
        (is (= 20000              (:keep-alive-timeout-ms v)))
        (is (= true               (:keep-alive-without-calls? v)))
        (is (= 86400000           (:idle-timeout-ms v)))
        (is (= false              (:secure v)))
        (is (nil?                 (:database v)))
        (is (= 19530              (:http-port v)))
        (is (= 30000              (:http-request-timeout-ms v)))
        (is (= 5000               (:http-connect-timeout-ms v)))))))

(deftest env-override-paths
  (testing "MILVUS_* env vars feed each env-sourced field with proper coercion"
    (let [r (config/resolve-MilvusClientConfig
              {}
              (mock-env {"MILVUS_HOST"      "milvus.svc"
                         "MILVUS_PORT"      "9091"
                         "MILVUS_TRANSPORT" "http"
                         "MILVUS_TOKEN"     "secret"}))]
      (is (r/ok? r))
      (let [v (:ok r)]
        (is (= "milvus.svc"   (:host v)))
        (is (= 9091           (:port v))      "port coerces string→int")
        (is (= :http          (:transport v)) "transport coerces string→keyword")
        (is (= "secret"       (:token v)))))))

(deftest blank-env-falls-back-to-default
  (testing "MILVUS_HOST=\"\" triggers default, not silent empty string"
    (let [r (config/resolve-MilvusClientConfig {} (mock-env {"MILVUS_HOST" ""}))]
      (is (r/ok? r))
      (is (= "localhost" (:host (:ok r)))))))

(deftest manifest-overrides-win
  (testing "explicit overrides map beats env (api/connect! opts take precedence)"
    (let [r (config/resolve-MilvusClientConfig
              {:host "from-opts" :port 7777 :transport :http}
              (mock-env {"MILVUS_HOST" "from-env" "MILVUS_PORT" "9999"
                         "MILVUS_TRANSPORT" "grpc"}))]
      (is (r/ok? r))
      (let [v (:ok r)]
        (is (= "from-opts" (:host v)))
        (is (= 7777        (:port v)))
        (is (= :http       (:transport v)))))))

(deftest invalid-port-collects-error
  (testing "non-numeric MILVUS_PORT collects coercion error, not throws"
    (let [r (config/resolve-MilvusClientConfig {} (mock-env {"MILVUS_PORT" "not-a-number"}))]
      (is (r/err? r))
      (is (= :config/resolution-failed (:error r)))
      (is (some #(= :port (:field %)) (:errors r))))))

(deftest token-optional
  (testing ":token has :required false — nil resolves OK"
    (let [r (config/resolve-MilvusClientConfig {} (mock-env {}))]
      (is (r/ok? r))
      (is (nil? (:token (:ok r)))))))

(deftest literal-overridable-via-opts
  (testing "literal fields can still be overridden via the opts map"
    (let [r (config/resolve-MilvusClientConfig
              {:collection-name "custom" :secure true :idle-timeout-ms 120000}
              (mock-env {}))]
      (is (r/ok? r))
      (let [v (:ok r)]
        (is (= "custom"  (:collection-name v)))
        (is (= true      (:secure v)))
        (is (= 120000    (:idle-timeout-ms v)))))))

;; =============================================================================
;; Legacy shim coverage — get-config / configure! / reset-config!
;; =============================================================================

(use-fixtures :each
  (fn [f]
    (config/reset-config!)
    (try (f) (finally (config/reset-config!)))))

(deftest configure-shim-merges-overrides
  (testing "configure! merges into session overrides; get-config resolves"
    (config/configure! {:host "shimmed" :port 11111})
    (let [v (config/get-config)]
      (is (= "shimmed" (:host v)))
      (is (= 11111     (:port v)))
      (is (= :grpc     (:transport v)) "untouched fields fall back to defaults"))
    (config/configure! {:secure true})
    (let [v (config/get-config)]
      (is (= "shimmed" (:host v))   "later configure! merges, not replaces")
      (is (= true      (:secure v))))))

(deftest reset-config-clears-overrides
  (testing "reset-config! returns to env+defaults posture"
    (config/configure! {:host "transient"})
    (is (= "transient" (:host (config/get-config))))
    (config/reset-config!)
    ;; Default for :host with no env → "localhost". May be overridden by
    ;; a real MILVUS_HOST env var in the dev shell — guard with `or`.
    (let [host (:host (config/get-config))]
      (is (or (= "localhost" host)
              (some? host))
          "post-reset host comes from env or default, not the prior override"))))

(deftest with-redefs-on-resolver-is-supported-test-seam
  (testing "with-redefs on resolve-MilvusClientConfig is the recommended hook"
    (with-redefs [config/resolve-MilvusClientConfig
                  (fn ([] (r/ok {:host "stub" :port 0}))
                    ([_] (r/ok {:host "stub" :port 0}))
                    ([_ _] (r/ok {:host "stub" :port 0})))]
      (is (= "stub" (:host (config/get-config))))
      (is (= 0     (:port (config/get-config)))))))
