/**
 * The AI-parsing provider table, and what the per-user form does with it.
 *
 * Pure and React-free on purpose: the admin hook next door drives the config form
 * with it, and both per-user Settings sections read `personalLlmProvider` out of it
 * without pulling in that hook (and, through it, the admin API module and the toast
 * context) just to resolve one stored value.
 */

export const LLM_MASKED = '••••••••'

/**
 * What the admin panel needs to know about each provider, in one table instead of a
 * chain of `provider === 'x' ?` ternaries spread over two shells. `selfHosted` is the
 * server-side distinction (see server llm-config.ts): a server the operator runs,
 * reached at an address they name, listing the models it has installed.
 */
export interface LlmProviderMeta {
  value: string
  label: string
  /** Short name of the software behind a self-hosted provider, shown as a chip. */
  badge?: string
  /** Default endpoint, shown as the Base URL placeholder. Absent = no Base URL field. */
  defaultBaseUrl?: string
  /** Placeholder for the Model field — a real model id for that provider. */
  modelPlaceholder: string
  /** Whether this provider names a server the operator runs (model list, optional key). */
  selfHosted: boolean
  /** Whether the server can download a model on request (Ollama can; LM Studio cannot). */
  canPull: boolean
}

export const LLM_PROVIDER_META: LlmProviderMeta[] = [
  {
    value: 'local',
    label: 'Local · Ollama',
    badge: 'Ollama',
    defaultBaseUrl: 'http://localhost:11434/v1',
    modelPlaceholder: 'select or pull below',
    selfHosted: true,
    canPull: true,
  },
  {
    value: 'lmstudio',
    label: 'Local · LM Studio',
    badge: 'LM Studio',
    defaultBaseUrl: 'http://localhost:1234/v1',
    modelPlaceholder: 'select below',
    selfHosted: true,
    // LM Studio downloads models in its own app / via `lms get`; its REST API has
    // no download endpoint, so the panel lists what is there and offers no Pull.
    canPull: false,
  },
  {
    value: 'openai',
    label: 'OpenAI',
    defaultBaseUrl: 'https://api.openai.com/v1',
    modelPlaceholder: 'gpt-4o',
    selfHosted: false,
    canPull: false,
  },
  {
    value: 'anthropic',
    label: 'Anthropic',
    modelPlaceholder: 'claude-opus-4-8',
    selfHosted: false,
    canPull: false,
  },
]

/** Curated models the local extractor is tuned for, pullable via Ollama. The router drives
 *  one model per document via the server's schema-enforced sampling; "thinking" is disabled
 *  automatically, so the Qwen3 family works without any tuning. A host only needs one. */
export const RECOMMENDED_MODELS: { id: string; label: string; note: string; recommended: boolean }[] = [
  { id: 'qwen3:8b', label: 'Qwen3 — 8B', note: 'Recommended · best extraction quality & speed on CPU (thinking auto-disabled) · Apache-2.0', recommended: true },
]

/**
 * What the per-user AI-parsing form shows for a stored provider.
 *
 * A self-hosted provider ('local' = Ollama, 'lmstudio' = LM Studio) names an address,
 * which is instance configuration an admin sets on the addon (#1772) — the server
 * answers 403 to a personal write of one. So a row left over from before that rule
 * (or written by an admin) displays as OpenAI rather than as a value the form would
 * only be refused for. Local state only: nothing is saved until Save is pressed.
 */
export function personalLlmProvider<T extends string>(stored: T): T | 'openai' {
  return LLM_PROVIDER_META.find(m => m.value === stored)?.selfHosted ? 'openai' : stored
}
