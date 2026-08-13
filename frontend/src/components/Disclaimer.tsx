import { motion } from 'framer-motion'
import { AlertTriangle, Calculator, ShieldAlert } from 'lucide-react'

export function Disclaimer({ utilityModel }: { utilityModel?: { label_0_utility: number; label_1_utility: number } }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="rounded-2xl border border-amber-400/25 bg-gradient-to-br from-amber-400/8 via-ink-850 to-ink-900 p-6 sm:p-8"
    >
      <div className="flex items-center gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-amber-400/25 bg-amber-400/10">
          <ShieldAlert className="h-5 w-5 text-amber-300" strokeWidth={1.9} />
        </div>
        <h2 className="font-display text-xl font-semibold text-slate-100">Important notice</h2>
      </div>

      <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="flex items-start gap-3 rounded-xl border border-amber-400/15 bg-ink-900/50 p-4">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" strokeWidth={2} />
          <p className="text-sm leading-relaxed text-amber-100/85">
            This system provides model-based treatment recommendations for research and educational purposes. It is not a
            clinical prescription or medical advice.
          </p>
        </div>
        <div className="flex items-start gap-3 rounded-xl border border-amber-400/15 bg-ink-900/50 p-4">
          <Calculator className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" strokeWidth={2} />
          <p className="text-sm leading-relaxed text-amber-100/85">
            The utility function is a simple model assumption
            {utilityModel
              ? ` (U(Label 0) = ${utilityModel.label_0_utility}, U(Label 1) = ${utilityModel.label_1_utility})`
              : ''}{' '}
            and is not a clinically validated utility scale.
          </p>
        </div>
      </div>

      <p className="mt-5 font-mono text-[11px] tracking-wide text-slate-500">
        ACTG175 · Final 23-edge Bayesian Network · Development-only parameter learning · BDeu prior · Variable
        Elimination
      </p>
    </motion.div>
  )
}
