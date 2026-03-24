import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { existsSync, readFileSync, writeFileSync, mkdirSync, appendFileSync } from "fs";
import { join, dirname } from "path";
import { homedir } from "os";
import { spawn, execSync, spawnSync } from "child_process";
import { fileURLToPath } from "url";

const PLUGIN_ID = "openclaw-null-embeddings";
/** Ollama's default API port — familiar for local setups; set `port` in plugin config if it conflicts. */
const DEFAULT_PORT = 11434;
const PROVIDER_NAME = "openai"; // pose as openai so the gateway's embedding resolver accepts us
const MODEL_ID = "null-embeddings-v1";

let serverChild: ReturnType<typeof spawn> | null = null;
let restartCount = 0;
let lastStartTime = 0;
const MAX_RESTARTS = 3;
const RESTART_DELAYS = [2000, 10000, 30000];
const STABLE_PERIOD = 300_000;

function getConfigPath(): string {
  return process.env.OPENCLAW_CONFIG_PATH || join(process.env.HOME || "", ".openclaw", "openclaw.json");
}

function getLogDir(): string {
  return process.env.OPENCLAW_LOGS_DIR || join(process.env.HOME || "", ".openclaw", "logs");
}

function parsePort(val: unknown): number {
  if (typeof val === "number" && Number.isFinite(val) && val > 0) return val;
  const n = parseInt(String(val), 10);
  return Number.isFinite(n) && n > 0 ? n : DEFAULT_PORT;
}

function killPid(pid: number): void {
  if (!Number.isFinite(pid) || pid <= 0) return;
  try {
    spawnSync("kill", ["-9", String(pid)], { stdio: "pipe", timeout: 5000 });
  } catch {}
}

function killPortProcess(port: number): void {
  // Try lsof first, fall back to /proc scan
  for (const cmd of [
    `lsof -ti :${port}`,
    `fuser ${port}/tcp 2>/dev/null`,
  ]) {
    try {
      const out = execSync(cmd, {
        encoding: "utf-8", timeout: 3000, stdio: ["pipe", "pipe", "pipe"],
      }).trim();
      if (out) {
        for (const token of out.split(/\s+/)) {
          const n = parseInt(token, 10);
          if (Number.isFinite(n) && n > 0) killPid(n);
        }
        return;
      }
    } catch {}
  }
}

function isServerRunning(port: number): boolean {
  try {
    const cmd = `node -e "fetch('http://127.0.0.1:${port}/health').then(r=>{process.exit(r.ok?0:1)}).catch(()=>process.exit(1))"`;
    execSync(cmd, { encoding: "utf-8", timeout: 5000, stdio: ["pipe", "pipe", "pipe"] });
    return true;
  } catch {
    return false;
  }
}

function configureMemorySearchProvider(port: number, logger: any): void {
  const configPath = getConfigPath();
  let config: Record<string, any> = {};
  try {
    config = JSON.parse(readFileSync(configPath, "utf-8"));
  } catch { return; }

  const agents = config.agents || {};
  const defaults = agents.defaults || {};
  const memorySearch = defaults.memorySearch || {};

  memorySearch.provider = PROVIDER_NAME;
  memorySearch.model = MODEL_ID;
  memorySearch.remote = {
    ...(memorySearch.remote || {}),
    baseUrl: `http://127.0.0.1:${port}/v1`,
    apiKey: "null-embeddings",
  };

  // Ensure FTS-weighted hybrid search
  if (!memorySearch.query) {
    memorySearch.query = { hybrid: { enabled: true, vectorWeight: 0, textWeight: 1 } };
  }
  if (!memorySearch.enabled) {
    memorySearch.enabled = true;
  }

  defaults.memorySearch = memorySearch;
  agents.defaults = defaults;
  config.agents = agents;

  writeFileSync(configPath, JSON.stringify(config, null, 2) + "\n");
  logger.info(`Configured memorySearch provider → http://127.0.0.1:${port}/v1`);
}

function startServer(opts: {
  pluginDir: string;
  port: number;
  logger: any;
}): void {
  const serverScript = join(opts.pluginDir, "server.mjs");
  if (!existsSync(serverScript)) {
    opts.logger.error(`Server script not found: ${serverScript}`);
    return;
  }

  // Another process (or a prior manual `node server.mjs`) may already own the port.
  if (isServerRunning(opts.port)) {
    opts.logger.info(
      `Null embeddings server already healthy on port ${opts.port} — not spawning a duplicate`,
    );
    return;
  }

  if (serverChild) {
    try { serverChild.kill("SIGKILL"); } catch {}
    serverChild = null;
  }
  killPortProcess(opts.port);

  // Wait for port to free
  for (let i = 0; i < 15; i++) {
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 200);
    if (!isServerRunning(opts.port)) break;
  }

  const logDir = getLogDir();
  try { mkdirSync(logDir, { recursive: true }); } catch {}
  const logPath = join(logDir, "null-embeddings.log");

  const child = spawn("node", [serverScript], {
    env: {
      PATH: process.env.PATH ?? "",
      HOME: process.env.HOME ?? "",
      NULL_EMBEDDINGS_PORT: String(opts.port),
    },
    stdio: ["ignore", "ignore", "pipe"],
  });
  serverChild = child;
  lastStartTime = Date.now();

  child.stderr?.on("data", (d: Buffer | string) => {
    const s = Buffer.isBuffer(d) ? d.toString("utf-8") : String(d);
    try { appendFileSync(logPath, s); } catch {}
  });

  child.on("error", (err: Error) => {
    opts.logger.error(`Null embeddings server error: ${err.message}`);
  });

  child.on("exit", (code, signal) => {
    opts.logger.info(`Null embeddings server exited (code ${code}, signal ${signal ?? "none"})`);
    if (serverChild === child) serverChild = null;
    if (code === 0 || code === null || signal != null) return;

    const uptime = Date.now() - lastStartTime;
    if (uptime > STABLE_PERIOD) restartCount = 0;
    if (restartCount >= MAX_RESTARTS) {
      opts.logger.error(`Null embeddings server crashed ${restartCount} times, not restarting`);
      return;
    }

    const delay = RESTART_DELAYS[Math.min(restartCount, RESTART_DELAYS.length - 1)];
    restartCount++;
    opts.logger.warn(`Null embeddings server crashed, restarting in ${delay / 1000}s (${restartCount}/${MAX_RESTARTS})`);
    setTimeout(() => startServer(opts), delay);
  });

  opts.logger.info(`Null embeddings server started on port ${opts.port} (pid ${child.pid})`);
}

function resolvePluginDir(api: OpenClawPluginApi): string {
  // Check install record first
  const installRecord = (api.config.plugins as any)?.installs?.[PLUGIN_ID];
  if (installRecord?.installPath && existsSync(join(installRecord.installPath, "server.mjs"))) {
    return installRecord.installPath;
  }
  // Convention path
  const conventionPath = join(homedir(), ".openclaw", "extensions", PLUGIN_ID);
  if (existsSync(join(conventionPath, "server.mjs"))) {
    return conventionPath;
  }
  // Fallback: directory of this file
  try {
    return dirname(fileURLToPath(import.meta.url));
  } catch {
    return api.resolvePath(".");
  }
}

const plugin = {
  id: PLUGIN_ID,
  name: "Null Embeddings",
  description:
    "Local OpenAI-compatible embeddings server that returns zero vectors. " +
    "Enables BM25/FTS memory indexing without a real embedding provider.",
  configSchema: {
    type: "object",
    properties: {
      port: {
        type: "number",
        default: DEFAULT_PORT,
        description: `Port for the null embeddings server. Default ${DEFAULT_PORT}.`,
      },
    },
  },

  register(api: OpenClawPluginApi) {
    const pluginDir = resolvePluginDir(api);
    const pluginConfig = api.pluginConfig || {};
    const port = parsePort(pluginConfig.port);
    const isPluginsInstall = process.argv.includes("plugins") && process.argv.includes("install");

    configureMemorySearchProvider(port, api.logger);

    // Avoid spawning during `openclaw memory *`, `plugins install`, etc. — those load
    // plugins but are not a long-lived gateway; spawning here fights cursor-brain for
    // ports and litters logs. Only start the child when the gateway actually comes up.
    if (!isPluginsInstall) {
      api.on("gateway_start", () => {
        startServer({ pluginDir, port, logger: api.logger });
      });
    }

    api.registerCli((ctx) => {
      const prog = ctx.program
        .command("null-embeddings")
        .description("Null Embeddings — zero-vector embedding server for BM25 memory");

      prog
        .command("status")
        .description("Show server status")
        .action(() => {
          const running = isServerRunning(port);
          console.log("Null Embeddings Status\n");
          console.log(`  Status:  ${running ? "running" : "stopped"}`);
          console.log(`  Port:    ${port}`);
          console.log(`  Model:   ${MODEL_ID}`);
          console.log(`  Log:     ${join(getLogDir(), "null-embeddings.log")}`);
        });

      prog
        .command("stop")
        .description("Stop the embeddings server")
        .action(async () => {
          if (!isServerRunning(port)) {
            console.log("Server is not running.");
            return;
          }
          killPortProcess(port);
          await new Promise((r) => setTimeout(r, 500));
          console.log(isServerRunning(port)
            ? `Server on port ${port} may still be running.`
            : `Server on port ${port} stopped.`);
        });

      prog
        .command("restart")
        .description("Restart the embeddings server (detached)")
        .action(async () => {
          const serverScript = join(pluginDir, "server.mjs");
          if (!existsSync(serverScript)) {
            console.error(`Server script not found: ${serverScript}`);
            process.exitCode = 1;
            return;
          }
          killPortProcess(port);
          await new Promise((r) => setTimeout(r, 500));
          const child = spawn("node", [serverScript], {
            env: {
              PATH: process.env.PATH ?? "",
              HOME: process.env.HOME ?? "",
              NULL_EMBEDDINGS_PORT: String(port),
            },
            stdio: "ignore",
            detached: true,
          });
          child.unref();
          console.log(`Server restarted on port ${port} (pid ${child.pid}).`);
        });
    }, { commands: ["null-embeddings"] });
  },
};

export default plugin;
