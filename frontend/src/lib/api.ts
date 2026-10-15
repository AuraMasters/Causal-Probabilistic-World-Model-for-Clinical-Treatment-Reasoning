import axios, { type AxiosInstance, type AxiosError } from 'axios'
import type { AnalyzeResult, Overview, PatientInputs } from './types'
import precomputedOverview from './precomputedOverview.json'
import { runClientInference } from './clientInference'

/**
 * Resolve the backend API base URL from Vite environment variables.
 * Falls back to relative '/api' for same-origin dev proxy or production deployment.
 */
const rawBaseUrl = import.meta.env.VITE_API_BASE_URL as string | undefined
export const API_BASE_URL = (rawBaseUrl ? rawBaseUrl.trim().replace(/\/+$/, '') : '') || '/api'

/**
 * Production-configured Axios client instance.
 */
export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 8000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
})

// Request Interceptor: Attach telemetry & trace headers
apiClient.interceptors.request.use(
  (config) => {
    config.headers['X-Client-Timestamp'] = new Date().toISOString()
    if (import.meta.env.DEV) {
      console.debug(`[API Request] ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`)
    }
    return config
  },
  (error) => Promise.reject(error),
)

// Response Interceptor: Format error messages from backend responses
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<{ error?: string; message?: string }>) => {
    let message = error.message
    if (error.response?.data?.error) {
      message = error.response.data.error
    } else if (error.response?.data?.message) {
      message = error.response.data.message
    } else if (error.code === 'ECONNABORTED') {
      message = 'Analysis request timed out after 8 seconds.'
    } else if (!error.response) {
      message = 'Unable to reach backend server. Verification or offline fallback active.'
    }
    return Promise.reject(new Error(message))
  },
)

/**
 * Health check endpoint to verify backend connectivity.
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const res = await apiClient.get<{ status: string }>('/health', { timeout: 2500 })
    return res.data?.status === 'ok'
  } catch {
    return false
  }
}

/**
 * Fetch overview, DAG structure, and validation benchmarks from the causal backend.
 * Falls back to precomputed artifacts if the server is offline.
 */
export async function fetchOverview(): Promise<Overview> {
  try {
    const response = await apiClient.get<Overview>('/overview')
    return response.data
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    console.info(`[Clinical AI] Backend overview unavailable (${message}). Using precomputed research artifacts.`)
    return precomputedOverview as unknown as Overview
  }
}

/**
 * Execute counterfactual patient analysis and treatment recommendation via backend SCM.
 * Seamlessly fails over to client-side inference if the server is unreachable.
 */
export async function analyzePatient(
  inputs: PatientInputs,
  modelType: 'continuous' | 'discretized' = 'continuous',
): Promise<AnalyzeResult> {
  try {
    const response = await apiClient.post<AnalyzeResult>('/analyze', {
      inputs,
      model_type: modelType,
    })
    return response.data
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    console.info(
      `[Clinical AI] Remote inference unavailable (${message}). Running offline ${
        modelType === 'discretized' ? 'discretized Bayesian network' : 'continuous SCM'
      } on client.`,
    )
    return runClientInference(inputs, modelType)
  }
}
