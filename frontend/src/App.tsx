import { AnimatePresence, motion } from 'framer-motion'
import {
  AlertCircle,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Database,
  GitBranch,
  RefreshCw,
  Sparkles,
  Stethoscope,
  UserCheck,
} from 'lucide-react'
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

const NAV_ITEMS = [
  { id: 'overview', label: 'Cohort & Model', icon: Database },
  { id: 'patient', label: 'Patient Evidence', icon: UserCheck },
  { id: 'recommendation', label: 'AI Decision', icon: Sparkles },
  { id: 'analysis', label: 'Comparative Arms', icon: BarChart3 },
  { id: 'network', label: 'Causal DAG', icon: GitBranch },
  { id: 'validation', label: 'Validation & Comparison', icon: CheckCircle2 },
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
  const [activeModel, setActiveModel] = useState<'continuous' | 'discretized'>('continuous')
  const [currentInputs, setCurrentInputs] = useState<PatientInputs>(DEFAULT_SAMPLE)
  const [activeSection, setActiveSection] = useState('overview')

  // Auto-run initial patient analysis once overview is loaded
  useEffect(() => {
    if (overview && state.status === 'idle') {
      void run(DEFAULT_SAMPLE, activeModel)
    }
  }, [overview, state.status, run, activeModel])

  const handleModelSelect = (model: 'continuous' | 'discretized') => {
    setActiveModel(model)
    void run(currentInputs, model)
  }

  const handlePatientAnalyze = (inputs: PatientInputs) => {
    setCurrentInputs(inputs)
    void run(inputs, activeModel)
  }

  // Active section scroll spy
  useEffect(() => {
    const sectionIds = NAV_ITEMS.map((item) => item.id)

    const handleScroll = () => {
      const scrollPos = window.scrollY + 180
      for (let i = sectionIds.length - 1; i >= 0; i--) {
        const el = document.getElementById(sectionIds[i])
        if (el && el.offsetTop <= scrollPos) {
          setActiveSection(sectionIds[i])
          break
        }
      }
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll()
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const scrollTo = (id: string) => {
    const el = document.getElementById(id)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  return (
    <div className="app-bg min-h-screen text-zinc-100 selection:bg-white selection:text-black">
      <Header activeModel={activeModel} onSelectModel={handleModelSelect} />

      {/* Sticky minimalist intelligence navigation */}
      <nav className="sticky top-0 z-50 border-b border-zinc-800 bg-black/90 backdrop-blur-xl shadow-lg shadow-black/50">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-2.5 sm:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-zinc-900 border border-zinc-800">
              <BrainCircuit className="h-4 w-4 text-zinc-200" />
            </div>
            <span className="hidden font-display text-sm font-bold tracking-tight text-white md:inline-block">
              ACTG175 Clinical World Model
            </span>
          </div>

          <div className="flex items-center gap-1 overflow-x-auto py-1 no-scrollbar sm:gap-1.5">
            {NAV_ITEMS.map((item) => {
              const isActive = activeSection === item.id
              const Icon = item.icon
              return (
                <button
                  key={item.id}
                  onClick={() => scrollTo(item.id)}
                  className={`inline-flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-1.5 font-mono text-xs font-semibold tracking-wide transition-all cursor-pointer ${
                    isActive
                      ? 'bg-white text-black shadow-sm'
                      : 'text-zinc-400 hover:text-white hover:bg-zinc-900'
                  }`}
                >
                  <Icon className={`h-3.5 w-3.5 ${isActive ? 'text-black' : 'text-zinc-400'}`} />
                  <span>{item.label}</span>
                </button>
              )
            })}
          </div>
        </div>
      </nav>

      <main className="mx-auto max-w-7xl px-4 pb-24 sm:px-8">
        <div className="space-y-16 pt-8 sm:pt-12">
          {/* SECTION 1: SYSTEM OVERVIEW */}
          <Section
            id="overview"
            eyebrow="Cohort & Architecture"
            title="System Overview"
            subtitle="The ACTG175 randomized clinical trial cohort and verified causal architectures for interventional treatment optimization."
          >
            {loading ? (
              <LoadingState label="Loading model overview and parameters..." />
            ) : error ? (
              <ErrorState message={error} onRetry={retry} />
            ) : overview ? (
              <OverviewCards overview={overview} />
            ) : null}
          </Section>

          {/* SECTION 2: PATIENT PROFILE (EVIDENCE) */}
          <Section
            id="patient"
            eyebrow="Clinical Evidence Input"
            title="Patient Clinical Profile"
            subtitle={
              activeModel === 'continuous'
                ? 'Numerical biomarkers (CD4, CD8, Karnofsky, Age, Weight) are preserved as exact continuous values without arbitrary binning.'
                : 'Baseline Model A maps continuous measurements to 3-bin quantile discrete evidence states.'
            }
          >
            {loading ? (
              <LoadingState label="Loading patient profile parameters..." />
            ) : overview ? (
              <PatientForm
                metadata={overview.discretization}
                analyzing={state.status === 'loading'}
                onAnalyze={handlePatientAnalyze}
              />
            ) : null}
          </Section>

          {/* SECTION 3: RECOMMENDED TREATMENT (AI DECISION) */}
          <Section
            id="recommendation"
            eyebrow="AI Interventional Recommendation"
            title="Optimal Regimen Selection"
            subtitle="Maximizes expected clinical utility under Pearlian causal graph mutilation do(trt = k)."
          >
            <AnimatePresence mode="wait">
              {state.status === 'loading' && (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <LoadingState label="Evaluating causal interventions across all 4 randomized treatment arms..." />
                </motion.div>
              )}

              {state.status === 'error' && (
                <motion.div
                  key="error"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <ErrorState message={state.message} />
                </motion.div>
              )}

              {state.status === 'success' && (
                <motion.div
                  key="success"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <RecommendedTreatment result={state.result} />
                </motion.div>
              )}
            </AnimatePresence>
          </Section>

          {/* SECTION 4: COMPARATIVE TREATMENT ANALYSIS */}
          <Section
            id="analysis"
            eyebrow="Interventional Analysis"
            title="Comparative Treatment Arms"
            subtitle="Predicted progression probabilities, progression-free survival, and expected utilities across all four arms."
          >
            {state.status === 'success' ? (
              <div className="space-y-10">
                <TreatmentRanking ranking={state.result.ranking} />
                <TreatmentComparison treatments={state.result.treatments} />
              </div>
            ) : (
              <div className="card-surface rounded-2xl p-12 text-center text-zinc-400">
                <Stethoscope className="mx-auto h-12 w-12 text-zinc-500 mb-3" />
                <p className="font-display text-base font-bold text-white">No Active Patient Inferences</p>
                <p className="text-xs text-zinc-400 mt-1 max-w-sm mx-auto">
                  Submit a clinical profile above to simulate interventional outcomes across all 4 treatment arms.
                </p>
              </div>
            )}
          </Section>

          {/* SECTION 5: INTERACTIVE CAUSAL DAG */}
          <Section
            id="network"
            eyebrow="Structural Causal Model"
            title="Bayesian Directed Acyclic Graph (DAG)"
            subtitle="The learned 23-edge causal network connecting demographic baseline factors, clinical biomarkers, treatment assignment, and primary endpoints."
          >
            {overview ? (
              <Suspense fallback={<LoadingState label="Rendering interactive Bayesian network..." />}>
                <DagGraph overview={overview} />
              </Suspense>
            ) : null}
          </Section>

          {/* SECTION 6: MODEL VALIDATION & COMPARISON */}
          <Section
            id="validation"
            eyebrow="Scientific Rigor & Validation"
            title="Model Validation & Side-by-Side Comparison"
            subtitle="Held-out test cohort evaluations (N = 428), Model A vs Model B benchmarks, calibration reliability, threshold sweeps, and Decision Curve Analysis."
          >
            {overview ? <ValidationSection overview={overview} /> : null}
          </Section>

          {/* SECTION 7: OUTCOME DEFINITION */}
          <OutcomeSection />
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800 bg-black py-10 text-center font-mono text-xs text-zinc-500">
        <div className="mx-auto max-w-7xl px-4 space-y-2">
          <p className="font-bold text-zinc-300">
            ACTG175 Clinical AI Decision Support System &middot; Causal World Model
          </p>
          <p className="text-[11px] text-zinc-500 max-w-2xl mx-auto">
            Research and educational decision-support prototype. Evaluated under strict 80/20 train/test partition without test set leakage.
          </p>
        </div>
      </footer>
    </div>
  )
}

function LoadingState({ label }: { label: string }) {
  return (
    <div className="card-surface flex flex-col items-center justify-center rounded-2xl py-16 text-center shadow-lg">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        className="h-9 w-9 rounded-full border-2 border-zinc-800 border-t-white mb-4"
      />
      <p className="font-display text-sm font-bold text-white">{label}</p>
      <p className="font-mono text-xs text-zinc-400 mt-1">Executing mathematical pipeline...</p>
    </div>
  )
}

function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950 p-8 text-center shadow-xl">
      <AlertCircle className="mx-auto h-10 w-10 text-zinc-400 mb-3" />
      <h3 className="font-display text-base font-bold text-white">Pipeline Execution Error</h3>
      <p className="font-mono text-xs text-zinc-400 mt-1 max-w-md mx-auto">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-5 inline-flex items-center gap-2 rounded-xl bg-zinc-900 border border-zinc-700 px-4 py-2 font-mono text-xs font-bold text-white hover:bg-zinc-800 transition-colors cursor-pointer"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          <span>Retry Connection</span>
        </button>
      )}
    </div>
  )
}
