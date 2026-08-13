import { motion } from 'framer-motion'
import { Activity, Network } from 'lucide-react'

export function Header() {
  return (
    <motion.header
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className="app-bg app-grid relative overflow-hidden"
    >
      <div className="mx-auto max-w-7xl px-5 pt-16 pb-14 sm:px-8">
        <div className="flex flex-col items-start gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-mint-400/25 bg-mint-400/10 px-3.5 py-1.5">
              <Activity className="h-3.5 w-3.5 text-mint-300" strokeWidth={2.2} />
              <span className="font-mono text-[11px] font-medium tracking-[0.2em] text-mint-300 uppercase">
                Research · Educational decision support
              </span>
            </div>
            <h1 className="font-display text-4xl font-bold tracking-tight text-slate-50 text-glow sm:text-5xl">
              Causal Clinical Reasoning
            </h1>
            <p className="mt-3 font-mono text-sm tracking-wide text-slate-400 sm:text-base">
              ACTG175 Bayesian Decision Support
            </p>
          </div>
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.25, duration: 0.5 }}
            className="glass flex items-center gap-3 rounded-2xl px-4 py-3"
          >
            <Network className="h-8 w-8 text-cyan-300" strokeWidth={1.6} />
            <div>
              <p className="text-sm font-semibold text-slate-100">Final DAG · 23 edges</p>
              <p className="font-mono text-xs text-slate-400">Causal model v1.0</p>
            </div>
          </motion.div>
        </div>
      </div>
    </motion.header>
  )
}
