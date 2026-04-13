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
  {:host                       (or (System/getenv "MILVUS_HOST") "localhost")
   :port                       (parse-long (or (System/getenv "MILVUS_PORT") "19530"))
   :collection-name            "hive-mcp-memory"
   :connect-timeout-ms         10000
   ;; Keepalive budget is tuned for fragile intermediaries (tailscale
   ;; userspace netstack, cloud NAT). 30 s ping is short enough that
   ;; gVisor won't consider the flow idle (default idle TCP window is
   ;; 60-120 s). keep-alive-without-calls? is the critical bit — without
   ;; it gRPC only pings during active RPCs, so idle clients die silently
   ;; and the first post-idle query sees "UNAVAILABLE: Keepalive failed".
   :keep-alive-time-ms         30000
   :keep-alive-timeout-ms      10000
   :keep-alive-without-calls?  true
   :idle-timeout-ms            86400000
   :secure                     false
   :token                      (System/getenv "MILVUS_TOKEN")
   :database                   nil})

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
