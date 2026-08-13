import { AnimatePresence, motion } from 'framer-motion'
import { AlertCircle, Database, Network, RefreshCw, Stethoscope } from 'lucide-react'
import { lazy, Suspense, useEffect, useState } from 'react'
import { Disclaimer } from './components/Disclaimer'
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

const DagGraph = lazy(() =>
  import('./components/DagGraph').then((module) => ({ default: module.DagGraph })),
)

const NAV_LINKS = [
  { href: '#input', label: 'Patient Input' },
  { href: '#decision', label: 'Decision' },
  { href: '#network', label: 'Bayesian Network' },
  { href: '#validation', label: 'Validation' },
]

export default function App() {
  const { overview, loading, error, retry } = useOverview()
  const { state, run } = useAnalyze()
  const [showNav, setShowNav] = useState(false)

  useEffect(() => {
    const onScroll = () => setShowNav(window.scrollY > 320)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <div className="app-bg min-h-screen text-slate-100">
      <Header />

      <AnimatePresence>
        {showNav && (
          <motion.nav
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            className="fixed inset-x-0 top-0 z-50 border-b border-slate-700/40 bg-ink-950/85 backdrop-blur-xl"
          >
            <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-3 sm:px-8">
              <p className="font-display text-sm font-semibold text-slate-100">Causal Clinical Reasoning</p>
              <div className="flex items-center gap-5">
                {NAV_LINKS.map((link) => (
                  <a
                    key={link.href}
                    href={link.href}
                    className="font-mono text-[11px] font-medium tracking-[0.14em] text-slate-400 uppercase transition-colors hover:text-mint-300"
                  >
                    {link.label}
                  </a>
                ))}
              </div>
            </div>
          </motion.nav>
        )}
      </AnimatePresence>

      <main className="mx-auto max-w-7xl px-5 pb-16 sm:px-8">
        <div className="space-y-16 pt-12">
          <Section
            id="overview"
            eyebrow="Dataset & model"
            title="Overview"
            subtitle="The ACTG175 trial cohort and the final causal Bayesian Network used for decision support."
          >
            {loading ? (
              <LoadingState label="Loading model overview…" />
            ) : error ? (
              <ErrorState message={error} onRetry={retry} />
            ) : overview ? (
              <OverviewCards overview={overview} />
            ) : null}
          </Section>

          <Section
            id="input"
            eyebrow="Patient profile"
            title="Patient input"
            subtitle="Enter the patient's clinical characteristics. Numerical values are converted to the model's discretized states; all four treatments are then evaluated automatically."
          >
            {overview ? (
              <PatientForm
                metadata={overview.discretization}
                analyzing={state.status === 'loading'}
                onAnalyze={run}
              />
            ) : (
              <LoadingState label="Loading model metadata…" />
            )}
          </Section>

          <AnimatePresence>
            {state.status === 'error' && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="flex items-start gap-3 rounded-2xl border border-rose-400/30 bg-rose-400/10 p-5"
              >
                <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-rose-risk" />
                <div>
                  <p className="text-sm font-semibold text-rose-100">Analysis failed</p>
                  <p className="mt-0.5 text-sm text-rose-200/80">{state.message}</p>
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
                  <h3 className="mb-3 font-display text-lg font-semibold text-slate-100">Outcome probability</h3>
                  <OutcomeSection result={state.result} />
                </div>
                <div>
                  <h3 className="mb-3 font-display text-lg font-semibold text-slate-100">Treatment ranking</h3>
                  <TreatmentRanking ranking={state.result.ranking} />
                </div>
              </div>

              <div>
                <h3 className="mb-3 font-display text-lg font-semibold text-slate-100">Treatment comparison</h3>
                <div className="card-surface rounded-2xl p-2">
                  <TreatmentComparison treatments={state.result.treatments} />
                </div>
              </div>
            </motion.div>
          )}

          <Section
            id="network"
            eyebrow="Causal structure"
            title="Bayesian Network"
            subtitle="The final 23-edge DAG learned from the ACTG175 development data. Click a node to inspect its model states, hover to highlight dependencies."
          >
            {overview ? (
              <Suspense fallback={<LoadingState label="Loading network…" />}>
                <DagGraph overview={overview} />
              </Suspense>
            ) : (
              <LoadingState label="Loading network…" />
            )}
          </Section>

          <Section
            id="validation"
            eyebrow="Model validation"
            title="Validation"
            subtitle="Hold-out evaluation on the 428 test patients (parameters learned on development data only)."
          >
            {overview ? <ValidationSection overview={overview} /> : <LoadingState label="Loading validation…" />}
          </Section>

          <Disclaimer utilityModel={overview?.utility_model} />
        </div>
      </main>

      <footer className="border-t border-slate-700/30 py-8">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-3 px-5 sm:flex-row sm:px-8">
          <p className="flex items-center gap-2 font-mono text-xs text-slate-500">
            <Stethoscope className="h-3.5 w-3.5" />
            Causal Clinical Reasoning · ACTG175 Bayesian Decision Support
          </p>
          <p className="flex items-center gap-2 font-mono text-[11px] text-slate-600">
            <Database className="h-3 w-3" /> 2,139 patients
            <Network className="h-3 w-3" /> 23-edge DAG
          </p>
        </div>
      </footer>
    </div>
  )
}

function LoadingState({ label }: { label: string }) {
  return (
    <div className="flex items-center justify-center gap-3 rounded-2xl border border-slate-700/30 bg-ink-900/50 py-10">
      <motion.span
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 1.1, ease: 'linear' }}
        className="h-5 w-5 rounded-full border-2 border-slate-600 border-t-mint-400"
      />
      <p className="text-sm text-slate-400">{label}</p>
    </div>
  )
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-2xl border border-rose-400/30 bg-rose-400/10 px-6 py-10 text-center">
      <AlertCircle className="h-6 w-6 text-rose-risk" />
      <p className="max-w-xl text-sm text-rose-100">{message}</p>
      <p className="text-xs text-rose-200/70">
        Make sure the analysis server is running (Flask on port 5000) and reload the page.
      </p>
      <button
        onClick={onRetry}
        className="inline-flex items-center gap-2 rounded-lg border border-rose-400/30 bg-rose-400/10 px-4 py-2 text-sm font-medium text-rose-100 transition-colors hover:bg-rose-400/20"
      >
        <RefreshCw className="h-4 w-4" />
        Retry
      </button>
    </div>
  )
}
