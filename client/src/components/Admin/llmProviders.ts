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
  /**
   * Whether this provider names a server the operator runs. Such a server lists the
   * models it has and downloads new ones on request, so this one flag gates the whole
   * model-management block — Ollama through `/api/pull`, LM Studio through its native
   * v1 REST API. It also decides whether the API-key field is optional.
   */
  selfHosted: boolean
}

export const LLM_PROVIDER_META: LlmProviderMeta[] = [
  {
    value: 'local',
    label: 'Local · Ollama',
    badge: 'Ollama',
    defaultBaseUrl: 'http://localhost:11434/v1',
    modelPlaceholder: 'select or pull below',
    selfHosted: true,
  },
  {
    value: 'lmstudio',
    label: 'Local · LM Studio',
    badge: 'LM Studio',
    defaultBaseUrl: 'http://localhost:1234/v1',
    modelPlaceholder: 'select or pull below',
    selfHosted: true,
  },
  {
    value: 'openai',
    label: 'OpenAI',
    defaultBaseUrl: 'https://api.openai.com/v1',
    modelPlaceholder: 'gpt-4o',
    selfHosted: false,
  },
  {
    value: 'anthropic',
    label: 'Anthropic',
    modelPlaceholder: 'claude-opus-4-8',
    selfHosted: false,
  },
]

/**
 * Curated models the local extractor is tuned for, downloadable on either self-hosted
 * server. The router drives one model per document via the server's schema-enforced
 * sampling; "thinking" is disabled automatically, so the Qwen3 family works without any
 * tuning. A host only needs one.
 *
 * The id is per server because the two name the same weights differently: Ollama uses a
 * tag (`qwen3.5:4b`), LM Studio a catalog identifier (`qwen/qwen3.5-4b`), and each only
 * downloads the spelling it knows. One row per model with an id per server keeps that a
 * lookup rather than a second list to keep in step.
 */
export interface RecommendedModel {
  /** Model id per provider value, as that server lists and downloads it. */
  ids: Record<string, string>
  label: string
  note: string
  recommended: boolean
  vision: boolean
}

export const RECOMMENDED_MODELS: RecommendedModel[] = [
  {
    ids: { local: 'qwen3.5:4b', lmstudio: 'qwen/qwen3.5-4b' },
    label: 'Qwen3.5 — 4B',
    note: 'Recommended · small and quick on CPU, 3.4 GB download, 256K context (thinking auto-disabled) · Apache-2.0',
    recommended: true,
    vision: true,
  },
]

/** One flat row per recommended model the given server knows, carrying ITS id for it. */
export function recommendedModelsFor(provider: string): (Omit<RecommendedModel, 'ids'> & { id: string })[] {
  return RECOMMENDED_MODELS.flatMap(({ ids, ...rest }) => (ids[provider] ? [{ ...rest, id: ids[provider] }] : []))
}

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
