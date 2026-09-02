import { motion } from 'framer-motion'
import { Activity, GitMerge, Sparkles } from 'lucide-react'

export function Header() {
  return (
    <header className="relative w-full border-b border-cyan-400/20 bg-ink-950 overflow-hidden">
      {/* Background atmosphere */}
      <div className="pointer-events-none absolute inset-0 app-bg" />
      <div className="pointer-events-none absolute inset-0 app-grid opacity-60" />
      <div className="pointer-events-none absolute -top-24 left-1/4 h-72 w-96 rounded-full bg-cyan-400/15 blur-3xl" />
      <div className="pointer-events-none absolute -top-20 right-1/4 h-64 w-80 rounded-full bg-mint-300/10 blur-3xl" />

      {/* Hero content */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: 'easeOut' }}
        className="relative mx-auto max-w-7xl px-5 pt-12 pb-12 sm:px-8 sm:pt-16 sm:pb-14"
      >
        <div className="flex flex-col items-start gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="max-w-3xl">
            <div className="mb-3.5 inline-flex items-center gap-2 rounded-full border border-cyan-400/35 bg-cyan-400/10 px-3.5 py-1.5 backdrop-blur-md shadow-sm shadow-cyan-500/10">
              <Activity className="h-3.5 w-3.5 text-cyan-300" strokeWidth={2.4} />
              <span className="font-mono text-[11px] font-bold tracking-[0.2em] text-mint-200 uppercase">
                Clinical AI &middot; Decision Support
              </span>
            </div>
            
            <h1 className="font-display text-4xl font-extrabold tracking-tight sm:text-5xl lg:text-6xl text-gradient-title leading-tight">
              Causal Clinical Reasoning
            </h1>
            
            <p className="mt-3.5 max-w-2xl font-sans text-sm font-medium tracking-wide text-cyan-200/80 sm:text-base">
              ACTG175 Pearlian Bayesian Decision Support &middot; Exact Interventional Inference
            </p>
          </div>

          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.15, duration: 0.4 }}
            className="glass flex items-center gap-3.5 rounded-2xl px-5 py-3.5 shadow-lg border border-cyan-400/25 shrink-0"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-400/10 border border-cyan-400/30">
              <GitMerge className="h-5 w-5 text-cyan-300" strokeWidth={2} />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <p className="text-sm font-bold text-mint-200">Final DAG &middot; 23 Edges</p>
                <Sparkles className="h-3.5 w-3.5 text-cyan-300" />
              </div>
              <p className="font-mono text-xs text-cyan-300/70">17 Variables &middot; BDeu Prior</p>
            </div>
          </motion.div>
        </div>
      </motion.div>
    </header>
  )
}
