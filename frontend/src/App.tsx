import { AnimatePresence, motion } from 'framer-motion'
import { AlertCircle, Database, Network, RefreshCw, Sparkles, Stethoscope } from 'lucide-react'
import { lazy, Suspense, useEffect, useState } from 'react'
import { Header } from './components/Header'
import { OutcomeSection } from './components/OutcomeSection'
import { OverviewCards } from './components/OverviewCards'
import { PatientForm } from './components/PatientForm'
import { RecommendedTreatment } from './components/RecommendedTreatment'
import { Section } from './components/Section'
import { TreatmentComparison } from './components/TreatmentComparison'
import { TreatmentRanking } from './components/TreatmentRanking'
import { ValidationSection } from './components/ValidationSection'
import { useAnalyze, useOverview } from './hooks/useModel'
import type { PatientInputs } from './lib/types'

const DagGraph = lazy(() =>
  import('./components/DagGraph').then((module) => ({ default: module.DagGraph })),
)

const NAV_LINKS = [
  { href: '#input', label: 'Patient Profile' },
  { href: '#decision', label: 'Decision Support' },
  { href: '#network', label: 'Causal DAG' },
  { href: '#validation', label: 'Validation' },
]

const DEFAULT_SAMPLE: PatientInputs = {
  age: '35',
  wtkg: '72',
  karnof: '100',
  preanti: '0',
  cd40: '480',
  cd80: '820',
  hemo: '0',
  homo: '1',
  drugs: '0',
  oprior: '0',
  z30: '0',
  race: '0',
  gender: '1',
  strat: '1',
  symptom: '0',
}

export default function App() {
  const { overview, loading, error, retry } = useOverview()
  const { state, run } = useAnalyze()
  const [showNav, setShowNav] = useState(false)

  useEffect(() => {
    const onScroll = () => setShowNav(window.scrollY > 280)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // Auto-run initial patient analysis once overview is loaded
  useEffect(() => {
    if (overview && state.status === 'idle') {
      void run(DEFAULT_SAMPLE)
    }
  }, [overview, state.status, run])

  return (
    <div className="app-bg min-h-screen text-slate-100 selection:bg-mint-400 selection:text-ink-950">
      <Header />

      <AnimatePresence>
        {showNav && (
          <motion.nav
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            className="fixed inset-x-0 top-0 z-50 border-b border-slate-800/80 bg-ink-950/90 backdrop-blur-xl shadow-lg shadow-black/20"
          >
            <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-3 sm:px-8">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-mint-400 animate-pulse" />
                <p className="font-display text-sm font-bold text-slate-100">Causal Clinical Reasoning</p>
              </div>
              <div className="flex items-center gap-5">
                {NAV_LINKS.map((link) => (
                  <a
                    key={link.href}
                    href={link.href}
                    className="font-mono text-[11px] font-semibold tracking-[0.14em] text-slate-400 uppercase transition-colors hover:text-mint-300"
                  >
                    {link.label}
                  </a>
                ))}
              </div>
            </div>
          </motion.nav>
        )}
      </AnimatePresence>

      <main className="mx-auto max-w-7xl px-5 pb-20 sm:px-8">
        <div className="space-y-16 pt-10 sm:pt-12">
          <Section
            id="overview"
            eyebrow="Cohort & Structure"
            title="System Overview"
            subtitle="The ACTG175 randomized trial cohort and the final 23-edge causal Bayesian Network utilized for interventional decision support."
          >
            {loading ? (
              <LoadingState label="Loading model overview and learned CPT parameters..." />
            ) : error ? (
              <ErrorState message={error} onRetry={retry} />
            ) : overview ? (
              <OverviewCards overview={overview} />
            ) : null}
          </Section>

          <Section
            id="input"
            eyebrow="Patient Evidence"
            title="Patient Clinical Profile"
            subtitle="Specify patient biomarkers and history. Continuous variables are discretized using data-driven quantile boundaries."
          >
            {overview ? (
              <PatientForm
                metadata={overview.discretization}
                analyzing={state.status === 'loading'}
                onAnalyze={run}
              />
            ) : (
              <LoadingState label="Loading model metadata..." />
            )}
          </Section>

          <AnimatePresence>
            {state.status === 'error' && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="flex items-start gap-3 rounded-2xl border border-rose-400/40 bg-rose-400/10 p-5 shadow-lg"
              >
                <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-rose-risk" />
                <div>
                  <p className="text-sm font-bold text-rose-100">Analysis failed</p>
                  <p className="mt-0.5 text-sm text-rose-200/90">{state.message}</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {state.status === 'success' && (
            <motion.div
              id="decision"
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
              className="scroll-mt-24 space-y-8"
            >
              <RecommendedTreatment result={state.result} />

              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <div>
                  <h3 className="mb-3 font-display text-lg font-bold text-slate-100 flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-mint-300" />
                    Posterior Outcome Probabilities
                  </h3>
                  <OutcomeSection result={state.result} />
                </div>
                <div>
                  <h3 className="mb-3 font-display text-lg font-bold text-slate-100 flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-cyan-300" />
                    Treatment Arms Ranking
                  </h3>
                  <TreatmentRanking ranking={state.result.ranking} />
                </div>
              </div>

              <div>
                <h3 className="mb-3 font-display text-lg font-bold text-slate-100">
                  Comprehensive Treatment Arms Comparison
                </h3>
                <div className="card-surface rounded-2xl p-3 shadow-lg">
                  <TreatmentComparison treatments={state.result.treatments} />
                </div>
              </div>
            </motion.div>
          )}

          <Section
            id="network"
            eyebrow="Causal Architecture"
            title="Bayesian Network DAG"
            subtitle="The verified 23-edge directed acyclic graph learned from ACTG175 development data. Click any node to view state spaces and local dependencies."
          >
            {overview ? (
              <Suspense fallback={<LoadingState label="Rendering DAG network layout..." />}>
                <DagGraph overview={overview} />
              </Suspense>
            ) : (
              <LoadingState label="Loading network..." />
            )}
          </Section>

          <Section
            id="validation"
            eyebrow="Empirical Performance"
            title="Held-Out Model Validation"
            subtitle="Evaluation strictly on the 428 held-out test patients (parameters fitted exclusively on development data)."
          >
            {overview ? <ValidationSection overview={overview} /> : <LoadingState label="Loading validation metrics..." />}
          </Section>
        </div>
      </main>

      <footer className="border-t border-slate-800/80 bg-ink-950/60 py-8">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-3 px-5 sm:flex-row sm:px-8">
          <p className="flex items-center gap-2 font-mono text-xs text-slate-400">
            <Stethoscope className="h-4 w-4 text-mint-400" />
            Causal Clinical Reasoning &middot; ACTG175 Pearlian Decision Support
          </p>
          <p className="flex items-center gap-3 font-mono text-[11px] text-slate-500">
            <span className="flex items-center gap-1.5"><Database className="h-3 w-3 text-cyan-400" /> 2,139 Patients</span>
            <span className="flex items-center gap-1.5"><Network className="h-3 w-3 text-mint-400" /> 23 Edges / 17 Nodes</span>
          </p>
        </div>
      </footer>
    </div>
  )
}

function LoadingState({ label }: { label: string }) {
  return (
    <div className="flex items-center justify-center gap-3 rounded-2xl border border-slate-800/60 bg-ink-900/50 py-12 shadow-inner">
      <motion.span
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 1.1, ease: 'linear' }}
        className="h-5 w-5 rounded-full border-2 border-slate-600 border-t-mint-400"
      />
      <p className="text-sm font-medium text-slate-300">{label}</p>
    </div>
  )
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-2xl border border-rose-400/30 bg-rose-400/10 px-6 py-10 text-center shadow-lg">
      <AlertCircle className="h-6 w-6 text-rose-risk" />
      <p className="max-w-xl text-sm font-bold text-rose-100">{message}</p>
      <p className="text-xs text-rose-200/80">
        Ensure the backend Flask API is running on port 5000 and click retry.
      </p>
      <button
        onClick={onRetry}
        className="mt-2 inline-flex items-center gap-2 rounded-xl border border-rose-400/40 bg-rose-400/15 px-5 py-2.5 text-sm font-bold text-rose-100 transition-colors hover:bg-rose-400/25 cursor-pointer"
      >
        <RefreshCw className="h-4 w-4" />
        Retry Connection
      </button>
    </div>
  )
}
