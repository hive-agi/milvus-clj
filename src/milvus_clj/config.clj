(ns milvus-clj.config
  "Milvus connection configuration and defaults.

   Manages host, port, collection-name, and gRPC transport settings.
   Configuration is held in an atom and can be updated at runtime.

   Mirrors the pattern from hive-mcp.chroma.connection — atom-based
   config with configure!/reset-config! lifecycle.")

;; ---------------------------------------------------------------------------
;; Defaults
;; ---------------------------------------------------------------------------

(def ^:private default-config
  {:host                  "localhost"
   :port                  19530
   :collection-name       "hive-mcp-memory"
   :connect-timeout-ms    10000
   :keep-alive-time-ms    55000
   :keep-alive-timeout-ms 20000
   :idle-timeout-ms       86400000
   :secure                false
   :token                 nil
   :database              nil})

;; ---------------------------------------------------------------------------
;; State
;; ---------------------------------------------------------------------------

(defonce ^:private config (atom default-config))

;; ---------------------------------------------------------------------------
;; Public API
;; ---------------------------------------------------------------------------

(defn get-config
  "Get current Milvus configuration map."
  []
  @config)

(defn configure!
  "Merge new settings into the Milvus configuration.

   Example:
     (configure! {:host \"milvus.prod\" :port 19530 :collection-name \"memories\"})"
  [opts]
  (swap! config merge opts)
  @config)

(defn reset-config!
  "Reset configuration to defaults."
  []
  (reset! config default-config))
