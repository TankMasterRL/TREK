import { describe, it, expect } from 'vitest'
import { LLM_PROVIDER_META, llmBaseUrlForSave, llmProviderMeta, personalLlmProvider } from './llmProviders'

describe('llmProviders', () => {
  it('offers every provider the server accepts, Anthropic-compatible included', () => {
    expect(LLM_PROVIDER_META.map(m => m.value)).toEqual(['local', 'openai', 'anthropic', 'anthropic-compatible'])
  })

  it('gives the Anthropic-compatible provider a Base URL field; the official Anthropic one has none', () => {
    expect(llmProviderMeta('anthropic-compatible').baseUrlPlaceholder).toBeTruthy()
    expect(llmProviderMeta('anthropic').baseUrlPlaceholder).toBeUndefined()
  })

  it('falls back to the first row for an unknown provider', () => {
    expect(llmProviderMeta('nope').value).toBe('local')
  })

  it('saves a base URL only for a provider that has the field', () => {
    expect(llmBaseUrlForSave('anthropic-compatible', ' https://gw.example.com ')).toBe('https://gw.example.com')
    expect(llmBaseUrlForSave('local', 'http://ollama:11434/v1')).toBe('http://ollama:11434/v1')
    expect(llmBaseUrlForSave('anthropic', 'http://left-over')).toBe('')
  })

  it('shows a stored endpoint provider as its hosted counterpart in the personal form (#1772)', () => {
    expect(personalLlmProvider('local')).toBe('openai')
    expect(personalLlmProvider('anthropic-compatible')).toBe('anthropic')
    expect(personalLlmProvider('openai')).toBe('openai')
    expect(personalLlmProvider('anthropic')).toBe('anthropic')
  })
})
