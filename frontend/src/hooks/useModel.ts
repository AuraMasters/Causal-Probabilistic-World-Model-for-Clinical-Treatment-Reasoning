import { useCallback, useEffect, useState } from 'react'
import { analyzePatient, fetchOverview } from '../lib/api'
import type { AnalyzeResult, Overview, PatientInputs } from '../lib/types'

const MAX_ATTEMPTS = 8
const BASE_DELAY_MS = 1000

async function fetchOverviewWithRetry(): Promise<Overview> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await fetchOverview()
    } catch (cause) {
      if (attempt >= MAX_ATTEMPTS) throw cause
      const delay = BASE_DELAY_MS * 2 ** (attempt - 1)
      await new Promise((resolve) => setTimeout(resolve, delay))
    }
  }
}

export function useOverview() {
  const [overview, setOverview] = useState<Overview | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setOverview(await fetchOverviewWithRetry())
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  return { overview, loading, error, retry: load }
}

type AnalyzeState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; result: AnalyzeResult }
  | { status: 'error'; message: string }

export function useAnalyze() {
  const [state, setState] = useState<AnalyzeState>({ status: 'idle' })

  const run = useCallback(async (inputs: PatientInputs) => {
    setState({ status: 'loading' })
    try {
      const result = await analyzePatient(inputs)
      setState({ status: 'success', result })
    } catch (cause) {
      setState({
        status: 'error',
        message: cause instanceof Error ? cause.message : String(cause),
      })
    }
  }, [])

  const reset = useCallback(() => setState({ status: 'idle' }), [])

  return { state, run, reset }
}
