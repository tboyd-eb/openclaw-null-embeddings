# openclaw-null-embeddings

OpenClaw plugin plus tiny HTTP server that exposes an **OpenAI-compatible `/v1/embeddings`** API returning **fixed zero vectors**. Use it when you want **BM25 / full-text memory search** without a real embedding provider: the indexer gets a valid embedding payload, while hybrid search can keep **vector weight at 0**.

## Defaults

- Listens on **port 11434** by default (same as Ollama’s default API port). Override with plugin config `port` or env `NULL_EMBEDDINGS_PORT` if that conflicts.
- The plugin configures `agents.defaults.memorySearch` to use this server and spawns the child on **`gateway_start`** (not on one-off CLIs like `openclaw memory`).

## Install

Install as a path-based OpenClaw plugin pointing at this directory, enable it in `openclaw.json`, then restart the gateway.

## Run standalone

```bash
node server.mjs
# or
NULL_EMBEDDINGS_PORT=11434 node server.mjs
```

## License

MIT
