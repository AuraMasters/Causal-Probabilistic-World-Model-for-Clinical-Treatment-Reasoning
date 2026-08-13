import { motion } from 'framer-motion'
import { BarChart3, Gauge, Scale, ShieldCheck } from 'lucide-react'
import type { Overview } from '../lib/types'
import { formatPercent } from '../lib/format'

export function ValidationSection({ overview }: { overview: Overview }) {
  const metrics = overview.validation
  const decision = overview.treatment_decision_validation

  const cards = [
    { label: 'Log Loss', value: metrics.log_loss.toFixed(4), hint: 'Lower is better' },
    { label: 'Brier Score', value: metrics.brier_score.toFixed(4), hint: 'Lower is better' },
    { label: 'ROC-AUC', value: metrics.roc_auc.toFixed(4), hint: 'Higher is better' },
    { label: 'Accuracy', value: formatPercent(metrics.accuracy, 2), hint: 'Prediction agreement' },
    { label: 'ECE', value: metrics.ece.toFixed(4), hint: 'Calibration error' },
    { label: 'Test patients', value: String(metrics.test_patients), hint: 'Hold-out evaluation' },
  ]

  return (
    <div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {cards.map((card, index) => (
          <motion.div
            key={card.label}
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: index * 0.05 }}
            className="card-surface rounded-xl p-4"
          >
            <p className="text-[11px] font-medium tracking-wide text-slate-500 uppercase">{card.label}</p>
            <p className="mt-1.5 font-mono text-xl font-semibold text-slate-50">{card.value}</p>
            <p className="mt-0.5 text-[10px] text-slate-600">{card.hint}</p>
          </motion.div>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="mt-5 flex flex-col gap-4 rounded-2xl border border-cyan-400/25 bg-gradient-to-br from-cyan-400/10 via-ink-850 to-ink-900 p-6 sm:flex-row sm:items-center"
      >
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border border-cyan-400/25 bg-cyan-400/10">
          <Scale className="h-6 w-6 text-cyan-300" strokeWidth={1.8} />
        </div>
        <div className="flex-1">
          <p className="font-mono text-[11px] font-semibold tracking-[0.2em] text-cyan-300 uppercase">
            Treatment decision validation
          </p>
          <p className="mt-2 text-sm leading-relaxed text-slate-300">
            The model recommended a treatment with{' '}
            <span className="font-semibold text-slate-100">lower predicted P(Label 1)</span> than the observed
            treatment for{' '}
            <span className="font-semibold text-mint-300">
              {decision.better_count}/{decision.total}
            </span>{' '}
            of test patients ({formatPercent(decision.rate, 0)}).
          </p>
        </div>
        <div className="text-center sm:text-right">
          <p className="font-display text-3xl font-bold text-mint-300">{formatPercent(decision.rate, 0)}</p>
          <p className="text-[11px] text-slate-500">of test patients</p>
        </div>
      </motion.div>

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Callout icon={ShieldCheck} text="Parameters learned on development data only; test data used solely for final evaluation." />
        <Callout icon={Gauge} text="This is not treatment accuracy — it measures whether the model-selected treatment has lower predicted risk than the observed one." />
        <Callout icon={BarChart3} text="Final 23-edge DAG, BDeu parameter learning (ESS = 10), Variable Elimination inference." />
      </div>
    </div>
  )
}

function Callout({ icon: Icon, text }: { icon: typeof ShieldCheck; text: string }) {
  return (
    <div className="flex items-start gap-2.5 rounded-xl border border-slate-500/20 bg-ink-900/60 px-4 py-3">
      <Icon className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" strokeWidth={1.9} />
      <p className="text-xs leading-relaxed text-slate-400">{text}</p>
    </div>
  )
}
