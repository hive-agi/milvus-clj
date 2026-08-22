# milvus-clj

<!-- hive-badges -->

[![Clojars Project](https://img.shields.io/clojars/v/io.github.hive-agi/milvus-clj.svg)](https://clojars.org/io.github.hive-agi/milvus-clj)
[![cljdoc](https://cljdoc.org/badge/io.github.hive-agi/milvus-clj)](https://cljdoc.org/d/io.github.hive-agi/milvus-clj/CURRENT)
[![release](https://github.com/hive-agi/milvus-clj/actions/workflows/release.yml/badge.svg)](https://github.com/hive-agi/milvus-clj/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<!-- /hive-badges -->

**A transport-abstract Clojure client for [Milvus](https://milvus.io).** The
caller picks gRPC or HTTP/REST; the protocols, the query surface and the error
taxonomy are the same either way.

## Coordinates

```clojure
;; deps.edn
io.github.hive-agi/milvus-clj {:mvn/version "0.2.5"}
```

## Usage

```clojure
(require '[milvus-clj.client :as client])

(client/with-client [c {:transport :grpc
                        :host "milvus.prod" :port 19530
                        :token "root:pw"}]
  (client/-query c "memories" {:vector [0.1 0.2 …] :limit 10}))

(client/with-client [c {:transport :http
                        :host "milvus.prod" :port 9091
                        :token "root:pw"}]
  (client/-query c "memories" {:vector [0.1 0.2 …] :limit 10}))
```

`with-client` scopes an ephemeral client; `client/make` builds one directly and
dispatches on `:transport`. `client/probe!` checks reachability and
`client/classify-error` maps a transport failure onto a stable keyword, so
callers do not pattern-match on driver exception classes.

## Layout

| Namespace | Provides |
|---|---|
| `milvus-clj.client` | `IMilvusCore` / `IMilvusAdmin` / `IMilvusExtras` protocols, `make`, `with-client`, `classify-error` |
| `milvus-clj.transport.grpc` | gRPC transport over `io.milvus/milvus-sdk-java` |
| `milvus-clj.transport.http` | HTTP/REST transport for the Milvus v2.5.x API, via `java.net.http.HttpClient` |
| `milvus-clj.schema` | `FieldType` and `CollectionSchema` construction |
| `milvus-clj.index` | Index creation and management |
| `milvus-clj.config` | Typed config through `hive-di` `defconfig` — env resolution returning a `Result` |
| `milvus-clj.api` | Legacy flat API over a singleton client, kept so existing callers need not migrate at once |

The protocols are split by interface-segregation: a transport implements the
core data plane and opts into admin/extras. **Adding a third transport is a new
file under `milvus-clj.transport.*` plus one case arm in `make`** — nothing else
changes.

## Config

`milvus-clj.config` declares `MilvusClientConfig` via `hive-di`'s `defconfig`,
generating the field registry, a closed Malli schema, and a resolver that
returns a `Result`. Resolution order per env-sourced field: explicit override →
environment variable → blank-to-nil → typed default → coercion.

## License

MIT.
