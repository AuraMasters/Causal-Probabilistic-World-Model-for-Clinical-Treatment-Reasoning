import { motion } from 'framer-motion'
import { Activity, Layers, Sparkles } from 'lucide-react'

interface HeaderProps {
  activeModel?: 'continuous' | 'discretized'
  onSelectModel?: (model: 'continuous' | 'discretized') => void
}

export function Header({ activeModel = 'continuous', onSelectModel }: HeaderProps) {
  return (
    <header className="relative w-full border-b border-zinc-800 bg-black overflow-hidden">
      {/* Subtle minimalist grid */}
      <div className="pointer-events-none absolute inset-0 app-grid opacity-40" />

      {/* Hero content */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: 'easeOut' }}
        className="relative mx-auto max-w-7xl px-5 pt-10 pb-10 sm:px-8 sm:pt-14 sm:pb-12"
      >
        <div className="flex flex-col items-start gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="max-w-3xl">
            <div className="mb-3.5 inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900 px-3.5 py-1.5 shadow-sm">
              <Activity className="h-3.5 w-3.5 text-zinc-300" strokeWidth={2.4} />
              <span className="font-mono text-[11px] font-bold tracking-[0.2em] text-zinc-300 uppercase">
                Clinical AI &middot; Causal Decision Support
              </span>
            </div>

            <h1 className="font-display text-3xl font-extrabold tracking-tight sm:text-5xl lg:text-6xl text-white leading-tight">
              Causal Clinical Reasoning
            </h1>

            <p className="mt-2.5 max-w-2xl font-sans text-sm font-medium tracking-wide text-zinc-400 sm:text-base">
              ACTG175 Randomized Trial &middot; Continuous Biomarkers &middot; Counterfactual Treatment Optimization
            </p>
          </div>

          {/* Model Selector Pill */}
          {onSelectModel && (
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.15, duration: 0.4 }}
              className="flex flex-col gap-2 rounded-xl p-3 shadow-lg border border-zinc-800 bg-zinc-950/90 shrink-0"
            >
              <div className="flex items-center gap-1.5 text-xs font-mono font-bold text-zinc-400 px-1">
                <Layers className="h-3.5 w-3.5 text-zinc-300" />
                <span>Active Model Architecture:</span>
              </div>
              <div className="flex items-center gap-1.5 bg-zinc-900 p-1 rounded-lg border border-zinc-800">
                <button
                  type="button"
                  onClick={() => onSelectModel('continuous')}
                  className={`rounded-md px-3 py-1.5 font-mono text-xs font-bold transition-all cursor-pointer ${
                    activeModel === 'continuous'
                      ? 'bg-white text-black shadow-sm'
                      : 'text-zinc-400 hover:text-white'
                  }`}
                >
                  <span className="flex items-center gap-1">
                    <Sparkles className="h-3 w-3" /> Model B: Continuous (Primary)
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => onSelectModel('discretized')}
                  className={`rounded-md px-3 py-1.5 font-mono text-xs font-bold transition-all cursor-pointer ${
                    activeModel === 'discretized'
                      ? 'bg-zinc-200 text-black shadow-sm'
                      : 'text-zinc-400 hover:text-white'
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
