import { AnimatePresence, motion } from 'framer-motion'
import {
  AlertCircle,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Database,
  GitBranch,
  Network,
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
  { id: 'validation', label: 'Validation', icon: CheckCircle2 },
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
  const [activeSection, setActiveSection] = useState('overview')

  // Auto-run initial patient analysis once overview is loaded
  useEffect(() => {
    if (overview && state.status === 'idle') {
      void run(DEFAULT_SAMPLE)
    }
  }, [overview, state.status, run])

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
    <div className="app-bg min-h-screen text-slate-100 selection:bg-cyan-400 selection:text-ink-950">
      <Header />

      {/* Sticky intelligence navigation */}
      <nav className="sticky top-0 z-50 border-b border-cyan-400/15 bg-ink-950/85 backdrop-blur-xl shadow-lg shadow-black/30">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-2.5 sm:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-cyan-400/15 border border-cyan-400/30">
              <BrainCircuit className="h-4 w-4 text-cyan-300 animate-pulse" />
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
                      ? 'bg-cyan-400/20 text-cyan-200 border border-cyan-400/40 shadow-sm shadow-cyan-400/20'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-ink-850'
                  }`}
                >
                  <Icon className={`h-3.5 w-3.5 ${isActive ? 'text-cyan-300' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </button>
              )
            })}
          </div>
        </div>
      </nav>

      <main className="mx-auto max-w-7xl px-4 pb-24 sm:px-8">
        <div className="space-y-20 pt-8 sm:pt-12">
          {/* SECTION 1: SYSTEM OVERVIEW */}
          <Section
            id="overview"
            eyebrow="Cohort & Architecture"
            title="System Overview"
            subtitle="The ACTG175 randomized trial cohort and the verified 23-edge causal Bayesian Network utilized for interventional decision support."
          >
            {loading ? (
              <LoadingState label="Loading model overview and learned CPT parameters..." />
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
            subtitle="Input individual baseline biomarkers and patient history. Continuous biomarkers are automatically mapped to calibrated discrete evidence states."
          >
            {overview ? (
              <PatientForm
                metadata={overview.discretization}
                analyzing={state.status === 'loading'}
                onAnalyze={run}
              />
            ) : (
              <LoadingState label="Loading patient evidence schema..." />
            )}
          </Section>

          {/* ERROR DISPLAY */}
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
                  <p className="text-sm font-bold text-rose-100">Causal inference execution failed</p>
                  <p className="mt-0.5 text-sm text-rose-200/90">{state.message}</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* SECTION 3 & 4: DECISION SUPPORT & ANALYSIS */}
          {state.status === 'success' && (
            <>
              {/* PRIMARY AI DECISION HERO */}
              <Section
                id="recommendation"
                eyebrow="Pearlian Interventional Decision"
                title="AI Treatment Recommendation"
                subtitle="Calculated via Pearl's do-calculus graph mutilation across all four antiretroviral therapy arms."
              >
                <RecommendedTreatment result={state.result} />
              </Section>

              {/* COMPARATIVE ANALYSIS */}
              <Section
                id="analysis"
                eyebrow="Multi-Arm Evaluation"
                title="Treatment Arms Comparison & Ranking"
                subtitle="Expected utility, progression-free survival probability, and comparative risk metrics across all available drug regimens."
              >
                <div className="space-y-8">
                  <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                    <div>
                      <div className="mb-3 flex items-center gap-2">
                        <Sparkles className="h-4 w-4 text-cyan-300" />
                        <h3 className="font-display text-base font-bold text-white uppercase tracking-wider">
                          Posterior Outcome Probabilities
                        </h3>
                      </div>
                      <OutcomeSection result={state.result} />
                    </div>

                    <div>
                      <div className="mb-3 flex items-center gap-2">
                        <BarChart3 className="h-4 w-4 text-mint-300" />
                        <h3 className="font-display text-base font-bold text-white uppercase tracking-wider">
                          Regimen Decision Hierarchy
                        </h3>
                      </div>
                      <TreatmentRanking ranking={state.result.ranking} />
                    </div>
                  </div>

                  <div>
                    <h3 className="mb-3 font-display text-base font-bold text-white uppercase tracking-wider">
                      Comprehensive Multi-Arm Matrix
                    </h3>
                    <TreatmentComparison treatments={state.result.treatments} />
                  </div>
                </div>
              </Section>
            </>
          )}

          {/* SECTION 5: CAUSAL DAG GRAPH */}
          <Section
            id="network"
            eyebrow="Causal Architecture"
            title="Bayesian Network DAG"
            subtitle="The verified 23-edge directed acyclic graph learned from ACTG175 trial data with temporal constraints (Baseline → Treatment → Outcome)."
          >
            {overview ? (
              <Suspense fallback={<LoadingState label="Rendering DAG network graph layout..." />}>
                <DagGraph overview={overview} />
              </Suspense>
            ) : (
              <LoadingState label="Loading network..." />
            )}
          </Section>

          {/* SECTION 6: HELD-OUT VALIDATION */}
          <Section
            id="validation"
            eyebrow="Empirical Verification"
            title="Held-Out Model Validation"
            subtitle="Rigorous out-of-sample evaluation on 428 held-out test patients with parameters learned strictly on the development partition."
          >
            {overview ? <ValidationSection overview={overview} /> : <LoadingState label="Loading validation metrics..." />}
          </Section>
        </div>
      </main>

      <footer className="border-t border-cyan-400/15 bg-ink-950 py-10">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-4 sm:flex-row sm:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-cyan-400/10 border border-cyan-400/25">
              <Stethoscope className="h-4 w-4 text-cyan-300" />
            </div>
            <div>
              <p className="font-display text-sm font-bold text-slate-100">
                Causal Clinical Reasoning Platform
              </p>
              <p className="font-mono text-xs text-slate-400">
                ACTG175 Bayesian World Model &middot; Exact Variable Elimination
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 font-mono text-xs text-slate-400">
            <span className="flex items-center gap-1.5 rounded-lg border border-slate-700/50 bg-ink-900 px-3 py-1">
              <Database className="h-3.5 w-3.5 text-cyan-300" /> 2,139 Patients
            </span>
            <span className="flex items-center gap-1.5 rounded-lg border border-slate-700/50 bg-ink-900 px-3 py-1">
              <Network className="h-3.5 w-3.5 text-mint-300" /> 23 Edges &middot; 17 Nodes
            </span>
          </div>
        </div>
      </footer>
    </div>
  )
}

function LoadingState({ label }: { label: string }) {
  return (
    <div className="flex items-center justify-center gap-3 rounded-2xl border border-cyan-400/15 bg-ink-900/60 py-12 shadow-inner">
      <motion.span
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 1.1, ease: 'linear' }}
        className="h-5 w-5 rounded-full border-2 border-slate-700 border-t-cyan-400"
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
