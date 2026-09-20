import { describe, it, expect, vi, beforeEach } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { useLlmParsingConfig } from './useLlmParsingConfig'
import { LLM_PROVIDER_META, personalLlmProvider } from './llmProviders'

const { llmLocalModels, llmLocalPull, updateAddon, toastSuccess, toastError } = vi.hoisted(() => ({
  llmLocalModels: vi.fn(),
  llmLocalPull: vi.fn(),
  updateAddon: vi.fn(),
  toastSuccess: vi.fn(),
  toastError: vi.fn(),
}))
vi.mock('../../api/client', () => ({ adminApi: { llmLocalModels, llmLocalPull, updateAddon } }))
vi.mock('../shared/Toast', () => ({ useToast: () => ({ success: toastSuccess, error: toastError }) }))

const addon = (config: Record<string, unknown> = {}) => ({ id: 'llm_parsing', config })

beforeEach(() => {
  vi.clearAllMocks()
  llmLocalModels.mockResolvedValue({ models: [{ name: 'qwen3-8b', size: 0 }] })
  llmLocalPull.mockResolvedValue(undefined)
  updateAddon.mockResolvedValue({ success: true })
})

/**
 * The ONE logic path behind the AI-parsing config in both admin shells — the desktop
 * tile shelf and the phone sheet render their own markup over this.
 */
describe('LLM_PROVIDER_META', () => {
  it('describes each provider the server accepts, and only those', () => {
    // Mirrors LLM_PROVIDERS in the server's llm-config.ts — a provider shown here
    // that the server does not know would be saved and then silently resolve to null.
    expect(LLM_PROVIDER_META.map(m => m.value)).toEqual(['local', 'lmstudio', 'openai', 'anthropic'])
  })

  it('marks exactly the two local servers as self-hosted, and only Ollama as pullable', () => {
    expect(LLM_PROVIDER_META.filter(m => m.selfHosted).map(m => m.value)).toEqual(['local', 'lmstudio'])
    expect(LLM_PROVIDER_META.filter(m => m.canPull).map(m => m.value)).toEqual(['local'])
  })

  it('gives each local server its own default port', () => {
    const byValue = Object.fromEntries(LLM_PROVIDER_META.map(m => [m.value, m.defaultBaseUrl]))
    expect(byValue.local).toBe('http://localhost:11434/v1')
    expect(byValue.lmstudio).toBe('http://localhost:1234/v1')
    // Anthropic has no endpoint to name, so its Base URL field never appears.
    expect(byValue.anthropic).toBeUndefined()
  })
})

describe('personalLlmProvider', () => {
  it('shows a stored self-hosted provider as OpenAI in the per-user form (#1772)', () => {
    expect(personalLlmProvider('local')).toBe('openai')
    expect(personalLlmProvider('lmstudio')).toBe('openai')
  })

  it('leaves a hosted provider alone', () => {
    expect(personalLlmProvider('openai')).toBe('openai')
    expect(personalLlmProvider('anthropic')).toBe('anthropic')
  })
})

describe('useLlmParsingConfig', () => {
  it('lists models from the configured server, naming which one to ask', async () => {
    const { result } = renderHook(() => useLlmParsingConfig(addon({ provider: 'lmstudio', baseUrl: '' })))
    await waitFor(() => expect(result.current.installed).toEqual(['qwen3-8b']))
    expect(llmLocalModels).toHaveBeenCalledWith('http://localhost:1234/v1', 'lmstudio')
  })

  it('asks nothing of a cloud provider and clears a previous local list', async () => {
    const { result } = renderHook(() => useLlmParsingConfig(addon({ provider: 'local' })))
    await waitFor(() => expect(result.current.installed).toEqual(['qwen3-8b']))

    act(() => result.current.setProvider('openai'))
    await waitFor(() => expect(result.current.installed).toEqual([]))
    // Still just the one call — a cloud provider has no local server to query.
    expect(llmLocalModels).toHaveBeenCalledTimes(1)
  })

  it('surfaces an unreachable server as an error instead of a stale list', async () => {
    llmLocalModels.mockRejectedValue(new Error('Could not reach local LLM server at http://localhost:1234'))
    const { result } = renderHook(() => useLlmParsingConfig(addon({ provider: 'lmstudio' })))
    await waitFor(() => expect(result.current.modelsErr).toMatch(/Could not reach/))
    expect(result.current.installed).toEqual([])
  })

  it('passes the provider to a pull, so the server can refuse where there is no download API', async () => {
    const { result } = renderHook(() => useLlmParsingConfig(addon({ provider: 'local' })))
    await waitFor(() => expect(result.current.installed).toEqual(['qwen3-8b']))

    await act(async () => { await result.current.pull('qwen3:8b') })
    expect(llmLocalPull).toHaveBeenCalledWith('http://localhost:11434/v1', 'qwen3:8b', expect.any(Function), 'local')
    expect(result.current.model).toBe('qwen3:8b')
  })

  it('trims the model and keeps the endpoint of a provider that has one', async () => {
    const { result } = renderHook(() =>
      useLlmParsingConfig(addon({ provider: 'lmstudio', baseUrl: 'http://lms.lan:1234/v1', apiKey: '••••••••', multimodal: true })),
    )
    act(() => result.current.setModel('  qwen3-8b  '))
    await act(async () => { await result.current.save() })

    expect(updateAddon).toHaveBeenCalledWith('llm_parsing', {
      config: { provider: 'lmstudio', model: 'qwen3-8b', baseUrl: 'http://lms.lan:1234/v1', apiKey: '••••••••', multimodal: true },
    })
    expect(toastSuccess).toHaveBeenCalled()
  })

  it('drops a stale base URL when saving a provider that has no endpoint of its own', async () => {
    const { result } = renderHook(() => useLlmParsingConfig(addon({ provider: 'lmstudio', baseUrl: 'http://lms.lan:1234/v1' })))
    act(() => result.current.setProvider('anthropic'))
    await act(async () => { await result.current.save() })

    // The stale local endpoint must not ride along — it would hijack Anthropic's.
    expect(updateAddon.mock.calls[0][1].config.baseUrl).toBe('')
  })

  it('reports a failed save instead of leaving the form looking saved', async () => {
    updateAddon.mockRejectedValue(new Error('nope'))
    const { result } = renderHook(() => useLlmParsingConfig(addon({ provider: 'openai' })))
    await act(async () => { await result.current.save() })
    expect(toastError).toHaveBeenCalled()
    expect(result.current.saving).toBe(false)
  })
})
