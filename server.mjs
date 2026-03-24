#!/usr/bin/env node
/**
 * Null Embeddings Server
 *
 * OpenAI-compatible embeddings endpoint that returns zero vectors.
 * Designed to satisfy embedding provider requirements when only
 * BM25/FTS search is needed (no actual vector similarity).
 *
 * Standalone:  node server.mjs  (default port matches Ollama: 11434)
 * Via plugin:  spawned as child process by the OpenClaw extension
 */

import { createServer } from "node:http";

/** Default matches Ollama's API port; override with NULL_EMBEDDINGS_PORT if Ollama is also running. */
const PORT = parseInt(process.env.NULL_EMBEDDINGS_PORT || "11434", 10);
const DEFAULT_DIMS = 1536;
const MODEL_ID = "null-embeddings-v1";

function parseBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString()));
      } catch (e) {
        reject(e);
      }
    });
    req.on("error", reject);
  });
}

function zeroVector(dims) {
  return new Array(dims).fill(0);
}

function estimateTokens(text) {
  return Math.ceil(text.length / 4);
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  if (url.pathname === "/health" || url.pathname === "/v1/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok", model: MODEL_ID, port: PORT }));
    return;
  }

  if (req.method === "POST" && (url.pathname === "/v1/embeddings" || url.pathname === "/embeddings")) {
    let body;
    try {
      body = await parseBody(req);
    } catch {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: { message: "Invalid JSON body", type: "invalid_request_error" } }));
      return;
    }

    const inputs = Array.isArray(body.input) ? body.input : [body.input];
    const dims = body.dimensions || DEFAULT_DIMS;
    const vec = zeroVector(dims);

    let totalTokens = 0;
    const data = inputs.map((text, index) => {
      const tokens = typeof text === "string" ? estimateTokens(text) : 1;
      totalTokens += tokens;
      return { object: "embedding", index, embedding: vec };
    });

    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({
      object: "list",
      data,
      model: body.model || MODEL_ID,
      usage: { prompt_tokens: totalTokens, total_tokens: totalTokens },
    }));
    return;
  }

  if (req.method === "GET" && url.pathname === "/v1/models") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({
      object: "list",
      data: [{ id: MODEL_ID, object: "model", owned_by: "local" }],
    }));
    return;
  }

  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: { message: "Not found", type: "not_found" } }));
});

server.on("error", (err) => {
  if (err && err.code === "EADDRINUSE") {
    // Port already bound (e.g. manual server or another plugin instance). Exit cleanly so the
    // parent does not treat this as a crash loop.
    console.error(`[null-embeddings] Port ${PORT} already in use — another instance is serving`);
    process.exit(0);
    return;
  }
  console.error("[null-embeddings]", err);
  process.exit(1);
});

server.listen(PORT, "127.0.0.1", () => {
  const msg = `Null embeddings server listening on http://127.0.0.1:${PORT}`;
  if (process.send) {
    process.send({ type: "ready", port: PORT });
  }
  console.log(msg);
});

process.on("SIGTERM", () => {
  server.close(() => process.exit(0));
});
process.on("SIGINT", () => {
  server.close(() => process.exit(0));
});
