;; Copyright (C) 2026 Pedro Gomes Branquinho (BuddhiLW) <pedrogbranquinho@gmail.com>
;;
;; SPDX-License-Identifier: MIT

(ns milvus-clj.config
  "Typed config for milvus-clj — declared via hive-di defconfig.

   Replaces the historical `defonce config (atom default-config)` pattern.
   Generates (via the macro):

     MilvusClientConfig            — ADT (:config/resolved | :unresolved | :invalid)
     MilvusClientConfig-fields     — field registry (source of truth)
     MilvusClientConfig-schema     — Malli closed map schema
     resolve-MilvusClientConfig    — resolver fn (0/1/2 arity) returning Result

   Resolution order (per env-sourced field):
     1. Explicit overrides map (e.g. opts passed to api/connect!)
     2. Environment variable lookup (System/getenv by default)
     3. blank->nil normalization (\"\" → trigger default)
     4. Pre-typed default (skips coercion)
     5. hive-dsl.coerce on string env values

   Mirror of `hive-milvus.config/MilvusConfig` for the env-sourced surface
   (host/port/transport/token), augmented with `literal` fields carrying
   the legacy gRPC keepalive / idle / TLS / HTTP defaults that the api.clj
   shim forwards into client/make via flat->nested.

   Backward-compat shims:
     get-config / configure! / reset-config! survive as thin wrappers over
     a private resolved-cache atom so the legacy api.clj caller path keeps
     working without rewriting every connect! call. Tests that need to
     stub config should use `with-redefs` on `resolve-MilvusClientConfig`."
  (:require [hive-di.core :refer [defconfig env literal]]
            [hive-dsl.result :as r]))

;; ---------------------------------------------------------------------------
;; Defconfig — env + literal field declarations
;; ---------------------------------------------------------------------------

(defconfig MilvusClientConfig
  ;; --- env-sourced (4 original getenv calls migrated; :database
  ;; promoted from a nil literal to an env source because hive-di
  ;; literals must be non-nil) ---
  :host                       (env "MILVUS_HOST"
                                    :default "localhost"
                                    :type :string
                                    :doc "Milvus host (gRPC or HTTP gateway)")
  :port                       (env "MILVUS_PORT"
                                    :default 19530
                                    :type :int
                                    :doc "Milvus port (gRPC default 19530)")
  :token                      (env "MILVUS_TOKEN"
                                    :type :string
                                    :required false
                                    :doc "Auth token (Zilliz Cloud / RBAC). Optional.")
  ;; Transport selector — :grpc (legacy default) or :http.
  ;; PR-3 hybrid: callers can opt in to :http via overrides or env.
  ;; api.clj's flat->nested reads this and routes to the right nested
  ;; {:transport :grpc/:http} shape for client/make.
  :transport                  (env "MILVUS_TRANSPORT"
                                    :default :grpc
                                    :type :keyword
                                    :doc "Transport keyword: :grpc or :http")

  ;; --- literal fields (no env source historically; carried as defaults) ---
  :collection-name            (literal "hive-mcp-memory"
                                       :doc "Target Milvus collection name")
  :connect-timeout-ms         (literal 10000 :type :int
                                       :doc "gRPC channel connect timeout (ms)")
  ;; Keepalive budget tuned for tailscale userspace netstack (gVisor)
  ;; and cloud NAT. 10 s ping (down from 30 s after empirical evidence
  ;; that 30 s still loses races with gVisor's idle reaper). The
  ;; without-calls? flag is the critical one — without it gRPC only
  ;; pings during active RPCs, so idle clients die silently and the
  ;; first post-idle query sees \"UNAVAILABLE: Keepalive failed\".
  ;; If even 10 s leaks, switch the transport to :http via the
  ;; client/make factory — HTTP has no idle state to lose.
  :keep-alive-time-ms         (literal 10000 :type :int
                                       :doc "gRPC keepalive ping interval (ms)")
  :keep-alive-timeout-ms      (literal 20000 :type :int
                                       :doc "gRPC keepalive ack timeout (ms)")
  :keep-alive-without-calls?  (literal true :type :bool
                                       :doc "Send keepalive pings even with no active RPCs")
  :idle-timeout-ms            (literal 86400000 :type :int
                                       :doc "gRPC idle channel reap timeout (ms)")
  :secure                     (literal false :type :bool
                                       :doc "Use TLS/SSL for the connection")
  ;; :database is env-sourced (rather than literal) because it's optional
  ;; — `literal` requires a non-nil value, but the legacy default was nil
  ;; (= server default). Env source with :required false yields nil when
  ;; MILVUS_DATABASE is unset, preserving legacy semantics.
  :database                   (env "MILVUS_DATABASE"
                                    :type :string
                                    :required false
                                    :doc "Milvus database name (nil = server default)")
  ;; HTTP-specific defaults — only consulted when :transport is :http.
  ;; Milvus 2.5+ serves both gRPC AND REST on port 19530 via the proxy;
  ;; port 9091 is the metrics/health endpoint (NOT REST). Don't change
  ;; :http-port to 9091 — it will return 404 page not found on every call.
  :http-port                  (literal 19530 :type :int
                                       :doc "HTTP REST port (Milvus 2.5+ shares 19530 with gRPC)")
  :http-request-timeout-ms    (literal 30000 :type :int
                                       :doc "HTTP request timeout (ms)")
  :http-connect-timeout-ms    (literal 5000 :type :int
                                       :doc "HTTP connect timeout (ms)"))

;; ---------------------------------------------------------------------------
;; Backward-compat shims — thin wrappers over the resolver
;;
;; Production callers (api.clj/connect!) historically did:
;;     (when (seq opts) (config/configure! opts))
;;     (let [flat (config/get-config)] ...)
;; The atom served as a session-scoped merge target. We preserve that
;; ergonomic via a session atom, but the source of truth is now the
;; resolver — `with-redefs` on `resolve-MilvusClientConfig` is the
;; supported test seam.
;; ---------------------------------------------------------------------------

(defonce ^:private session-overrides (atom {}))

(defn- resolve-or-throw
  "Resolve current overrides → flat config map. Throws on resolution failure
   to preserve the historical fail-fast behaviour of `connect!`."
  [overrides]
  (let [result (resolve-MilvusClientConfig overrides)]
    (if (r/ok? result)
      (:ok result)
      (throw (ex-info (str "MilvusClientConfig resolution failed: "
                           (pr-str (:errors result)))
                      {:result result :overrides overrides})))))

(defn get-config
  "Get current Milvus configuration map.

   DEPRECATED: prefer calling `resolve-MilvusClientConfig` with explicit
   overrides at the api boundary. This shim resolves with whatever
   session-scoped overrides have been merged via `configure!`."
  []
  (resolve-or-throw @session-overrides))

(defn configure!
  "Merge new settings into the session-scoped Milvus overrides map.

   DEPRECATED: prefer passing opts directly to `api/connect!` (which
   forwards to `resolve-MilvusClientConfig`). Retained so existing
   one-arg `(connect! {...})` callers keep working without churn.

   Example:
     (configure! {:host \"milvus.prod\" :port 19530 :collection-name \"memories\"})"
  [opts]
  (swap! session-overrides merge opts)
  (resolve-or-throw @session-overrides))

(defn reset-config!
  "Reset session overrides to empty — subsequent `get-config` resolves
   purely from env + defconfig defaults."
  []
  (reset! session-overrides {}))
