import { existsSync } from 'node:fs'
import { createConnection } from 'node:net'
import os from 'node:os'
import path from 'node:path'
import { spawn, type ChildProcess } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import type { Plugin } from 'vite'

const API_PORT = 5000
const API_HOST = '127.0.0.1'
const POLL_INTERVAL_MS = 2000

const frontendDir = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(frontendDir, '..')
const pythonBin = path.join(repoRoot, '.venv-linux', 'bin', 'python')
const apiScript = path.join(repoRoot, 'src', 'api', 'app.py')

function portOpen(port: number, host = API_HOST): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = createConnection({ port, host })
    const done = (open: boolean) => {
      socket.destroy()
      resolve(open)
    }
    socket.setTimeout(1500, () => done(false))
    socket.once('connect', () => done(true))
    socket.once('error', () => done(false))
  })
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

/**
 * Dev-only plugin that spawns the Flask analysis API as a child of the Vite
 * dev server, so `npm run dev` runs the whole stack from one command.
 *
 * - Skips spawning if something already listens on :5000 (e.g. a manually
 *   started API).
 * - Warns (instead of failing) when the Linux venv is missing, so the proxy
 *   can still target a manually started server.
 * - Kills the Flask child when the dev server shuts down; no orphan listener.
 */
export function apiPlugin(): Plugin {
  let child: ChildProcess | null = null
  let shuttingDown = false
  let pollHandle: ReturnType<typeof setInterval> | null = null

  const killChild = () => {
    if (child && !child.killed) {
      child.kill()
    }
    child = null
  }

  const onExit = () => {
    shuttingDown = true
    if (pollHandle) clearInterval(pollHandle)
    killChild()
  }

  return {
    name: 'api-plugin',
    configureServer(server) {
      const waitForReady = async () => {
        await sleep(250)
        pollHandle = setInterval(async () => {
          if (shuttingDown) return
          if (await portOpen(API_PORT)) {
            if (pollHandle) clearInterval(pollHandle)
            console.log('[api] ready — dashboard live at http://localhost:5173')
          }
        }, POLL_INTERVAL_MS)
      }

      const startApi = async () => {
        if (await portOpen(API_PORT)) {
          console.log(`[api] server already listening on :${API_PORT} — not spawning a duplicate.`)
          await waitForReady()
          return
        }
        if (!existsSync(pythonBin)) {
          console.warn(
            `[api] ${pythonBin} not found. Start the API manually with scripts/run_api.sh, ` +
              'or configure a different Python/venv.',
          )
          await waitForReady()
          return
        }
        console.log('[api] starting Flask API (first load can take 10–20 s while the model imports)…')
        child = spawn(pythonBin, [apiScript], {
          cwd: repoRoot,
          env: {
            ...process.env,
            PYTHONPYCACHEPREFIX: process.env.PYTHONPYCACHEPREFIX ?? path.join(os.homedir(), '.pycache'),
          },
          stdio: ['ignore', 'pipe', 'pipe'],
        })
        child.stdout?.on('data', (chunk: Buffer) =>
          console.log(
            chunk
              .toString()
              .split('\n')
              .filter(Boolean)
              .map((line) => `[api] ${line}`)
              .join('\n'),
          ),
        )
        child.stderr?.on('data', (chunk: Buffer) =>
          console.error(
            chunk
              .toString()
              .split('\n')
              .filter(Boolean)
              .map((line) => `[api] ${line}`)
              .join('\n'),
          ),
        )
        child.on('exit', (code) => {
          child = null
          if (!shuttingDown) {
            console.error(`[api] Flask exited unexpectedly with code ${code ?? 'null'}.`)
          }
        })
        await waitForReady()
      }

      void startApi()

      const handleSignal = () => {
        onExit()
        process.exit(0)
      }

      server.httpServer?.once('close', onExit)
      process.once('SIGINT', handleSignal)
      process.once('SIGTERM', handleSignal)
    },
  }
}
