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
   ;; Keepalive budget tuned for tailscale userspace netstack (gVisor)
   ;; and cloud NAT. 10 s ping (down from 30 s after empirical evidence
   ;; that 30 s still loses races with gVisor's idle reaper). The
   ;; without-calls? flag is the critical one — without it gRPC only
   ;; pings during active RPCs, so idle clients die silently and the
   ;; first post-idle query sees "UNAVAILABLE: Keepalive failed".
   ;; If even 10 s leaks, switch the transport to :http via the
   ;; client/make factory — HTTP has no idle state to lose.
   :keep-alive-time-ms         10000
   :keep-alive-timeout-ms      20000
   :keep-alive-without-calls?  true
   :idle-timeout-ms            86400000
   :secure                     false
   :token                      (System/getenv "MILVUS_TOKEN")
   :database                   nil
   ;; Transport selector — :grpc (legacy default) or :http.
   ;; PR-3 hybrid: callers can opt in to :http via configure! or by
   ;; passing :transport in the connect! opts map. The api.clj shim
   ;; reads this and routes to the right nested {:transport :grpc/:http}
   ;; shape for client/make.
   :transport                  (if-let [t (System/getenv "MILVUS_TRANSPORT")]
                                 (keyword t)
                                 :grpc)
   ;; HTTP-specific defaults — only consulted when :transport is :http.
   :http-port                  9091
   :http-request-timeout-ms    30000
   :http-connect-timeout-ms    5000})

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
