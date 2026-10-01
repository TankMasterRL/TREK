/**
 * The AI-parsing provider table, and what the per-user form does with it.
 *
 * Pure and React-free on purpose: both admin shells (desktop AddonManager and the
 * mobile MAdminAddonManager) drive their config form from it instead of a chain of
 * `provider === 'x' ?` ternaries each, and both per-user Settings sections read
 * `personalLlmProvider` out of it. Icons stay in the shells — they are markup.
 */

export interface LlmProviderMeta {
  value: string
  label: string
  /** Short name of the software behind the provider, shown as a chip. */
  badge?: string
  /** Base URL placeholder. Absent = the provider has a fixed address and no Base URL field. */
  baseUrlPlaceholder?: string
  /** Placeholder for the Model field. */
  modelPlaceholder: string
  /** Placeholder for the API-key field. */
  apiKeyPlaceholder: string
  /** A caption shown under the connection fields, when the provider needs one. */
  hint?: string
}

export const LLM_PROVIDER_META: LlmProviderMeta[] = [
  {
    value: 'local',
    label: 'Local · OpenAI-compatible',
    badge: 'Ollama',
    baseUrlPlaceholder: 'http://localhost:11434/v1',
    modelPlaceholder: 'select or pull below',
    apiKeyPlaceholder: '(often not required)',
  },
  {
    value: 'openai',
    label: 'OpenAI',
    baseUrlPlaceholder: 'https://api.openai.com/v1',
    modelPlaceholder: 'gpt-4o',
    apiKeyPlaceholder: 'sk-…',
  },
  {
    value: 'anthropic',
    label: 'Anthropic',
    modelPlaceholder: 'claude-opus-4-8',
    apiKeyPlaceholder: 'sk-…',
    hint: 'Anthropic reads PDFs (including scans) natively. Local/OpenAI models receive extracted text — scanned PDFs need Anthropic.',
  },
  {
    value: 'anthropic-compatible',
    label: 'Anthropic-compatible',
    badge: 'Messages API',
    baseUrlPlaceholder: 'https://gateway.example.com',
    modelPlaceholder: 'model id the endpoint serves',
    apiKeyPlaceholder: 'API key for the endpoint',
    hint: 'Any server speaking the Anthropic Messages API (a LiteLLM or OpenRouter gateway, or another vendor’s Anthropic endpoint). The Base URL is required; requests go to {Base URL}/v1/messages. Documents are sent as extracted text.',
  },
]

/** The table row for a provider value, falling back to the first row for an unknown one. */
export function llmProviderMeta(value: string): LlmProviderMeta {
  return LLM_PROVIDER_META.find(m => m.value === value) ?? LLM_PROVIDER_META[0]
}

/**
 * The base URL the admin form saves for a provider: blank for a provider with a
 * fixed address (it has no field, so a value left over from another provider
 * must not ride along), the trimmed input otherwise.
 */
export function llmBaseUrlForSave(provider: string, baseUrl: string): string {
  return llmProviderMeta(provider).baseUrlPlaceholder ? baseUrl.trim() : ''
}

/**
 * What the per-user AI-parsing form shows for a stored provider.
 *
 * An endpoint provider names an address, which is instance configuration an admin
 * sets (#1772) — the server answers 403 to a personal write of one. So a row that
 * holds one (left over from before that rule, or inherited from the admin default)
 * displays as the hosted provider speaking the same API: 'local' as OpenAI,
 * 'anthropic-compatible' as Anthropic. Local state only: nothing is saved until
 * Save is pressed.
 */
export function personalLlmProvider<T extends string>(stored: T): T | 'openai' | 'anthropic' {
  if (stored === 'local') return 'openai'
  if (stored === 'anthropic-compatible') return 'anthropic'
  return stored
}
