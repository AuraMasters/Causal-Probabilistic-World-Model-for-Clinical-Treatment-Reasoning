import { motion } from 'framer-motion'
import { AlertTriangle, Calculator, ShieldAlert } from 'lucide-react'

export function Disclaimer({ utilityModel }: { utilityModel?: { label_0_utility: number; label_1_utility: number } }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="rounded-xl border border-zinc-800 bg-zinc-950 p-6 sm:p-8"
    >
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-zinc-800 bg-zinc-900 text-white">
          <ShieldAlert className="h-5 w-5 text-white" strokeWidth={1.8} />
        </div>
        <h3 className="font-display text-lg font-bold text-white">Important Notice</h3>
      </div>

      <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="flex items-start gap-3 rounded-lg border border-zinc-800 bg-zinc-900/60 p-4">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-zinc-300" strokeWidth={2} />
          <p className="text-sm leading-relaxed text-zinc-300">
            This system provides model-based treatment recommendations for research and educational purposes. It is not a
            clinical prescription or medical advice.
          </p>
        </div>
        <div className="flex items-start gap-3 rounded-lg border border-zinc-800 bg-zinc-900/60 p-4">
          <Calculator className="mt-0.5 h-4 w-4 shrink-0 text-zinc-300" strokeWidth={2} />
          <p className="text-sm leading-relaxed text-zinc-300">
            The utility function is a simple model assumption
            {utilityModel
              ? ` (U(Label 0) = ${utilityModel.label_0_utility}, U(Label 1) = ${utilityModel.label_1_utility})`
              : ''}{' '}
            and is not a clinically validated utility scale.
          </p>
        </div>
      </div>

      <p className="mt-5 font-mono text-[11px] tracking-wide text-zinc-500">
        ACTG175 · Final 23-edge Bayesian Network · Development-only parameter learning · BDeu prior · Variable
        Elimination
      </p>
    </motion.div>
  )
}
