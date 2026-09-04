import type { AnalyzeResult, Overview, PatientInputs } from './types'
import precomputedOverview from './precomputedOverview.json'
import { runClientInference } from './clientInference'

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) || '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 3500)

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      ...init,
    })

    let payload: unknown
    try {
      payload = await response.json()
    } catch {
      throw new Error(`Unexpected response from the analysis server (${response.status}).`)
    }

    if (!response.ok) {
      const message =
        payload && typeof payload === 'object' && 'error' in payload
          ? String((payload as { error: string }).error)
          : `Request failed with status ${response.status}.`
      throw new Error(message)
    }

    return payload as T
  } finally {
    clearTimeout(timeoutId)
  }
}

export async function fetchOverview(): Promise<Overview> {
  try {
    return await request<Overview>('/overview')
  } catch {
    console.info('[Clinical AI] Using bundled precomputed validation & benchmark artifacts.')
    return precomputedOverview as unknown as Overview
  }
}

export async function analyzePatient(
  inputs: PatientInputs,
  modelType: 'continuous' | 'discretized' = 'continuous',
): Promise<AnalyzeResult> {
  try {
    return await request<AnalyzeResult>('/analyze', {
      method: 'POST',
      body: JSON.stringify({ inputs, model_type: modelType }),
    })
  } catch {
    console.info('[Clinical AI] Running high-precision continuous causal inference client-side.')
    return runClientInference(inputs, modelType)
  }
}
