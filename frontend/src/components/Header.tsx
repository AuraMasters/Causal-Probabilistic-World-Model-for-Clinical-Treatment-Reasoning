import { motion } from 'framer-motion'
import { Activity, Layers, Sparkles } from 'lucide-react'

interface HeaderProps {
  activeModel?: 'continuous' | 'discretized'
  onSelectModel?: (model: 'continuous' | 'discretized') => void
}

export function Header({ activeModel = 'continuous', onSelectModel }: HeaderProps) {
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
        className="relative mx-auto max-w-7xl px-5 pt-10 pb-10 sm:px-8 sm:pt-14 sm:pb-12"
      >
        <div className="flex flex-col items-start gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="max-w-3xl">
            <div className="mb-3.5 inline-flex items-center gap-2 rounded-full border border-cyan-400/35 bg-cyan-400/10 px-3.5 py-1.5 backdrop-blur-md shadow-sm shadow-cyan-500/10">
              <Activity className="h-3.5 w-3.5 text-cyan-300" strokeWidth={2.4} />
              <span className="font-mono text-[11px] font-bold tracking-[0.2em] text-mint-200 uppercase">
                Clinical AI &middot; Causal Decision Support
              </span>
            </div>

            <h1 className="font-display text-3xl font-extrabold tracking-tight sm:text-5xl lg:text-6xl text-gradient-title leading-tight">
              Causal Clinical Reasoning
            </h1>

            <p className="mt-2.5 max-w-2xl font-sans text-sm font-medium tracking-wide text-cyan-200/80 sm:text-base">
              ACTG175 Randomized Trial &middot; Continuous Biomarkers &middot; Counterfactual Treatment Optimization
            </p>
          </div>

          {/* Model Selector Pill */}
          {onSelectModel && (
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.15, duration: 0.4 }}
              className="glass flex flex-col gap-2 rounded-2xl p-3 shadow-lg border border-cyan-400/25 shrink-0"
            >
              <div className="flex items-center gap-1.5 text-xs font-mono font-bold text-slate-300 px-1">
                <Layers className="h-3.5 w-3.5 text-cyan-300" />
                <span>Active Model Architecture:</span>
              </div>
              <div className="flex items-center gap-1.5 bg-ink-950 p-1 rounded-xl border border-slate-800">
                <button
                  type="button"
                  onClick={() => onSelectModel('continuous')}
                  className={`rounded-lg px-3 py-1.5 font-mono text-xs font-bold transition-all cursor-pointer ${
                    activeModel === 'continuous'
                      ? 'bg-gradient-to-r from-cyan-400 to-sky-500 text-ink-950 shadow-md shadow-cyan-400/20'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <span className="flex items-center gap-1">
                    <Sparkles className="h-3 w-3" /> Model B: Continuous (Primary)
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => onSelectModel('discretized')}
                  className={`rounded-lg px-3 py-1.5 font-mono text-xs font-bold transition-all cursor-pointer ${
                    activeModel === 'discretized'
                      ? 'bg-amber-400/25 text-amber-200 border border-amber-400/40 shadow-sm'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Model A: Discretized BN
                </button>
              </div>
            </motion.div>
          )}
        </div>
      </motion.div>
    </header>
  )
}
