import { maybe_encrypt_api_key, decrypt_api_key } from '../common/crypto/apiKeyCrypto';

/**
 * Shared types + helpers for the `llm_parsing` addon configuration.
 *
 * Config can live in two places (resolution happens in
 * server/src/nest/llm-parse/llm-config.resolver.ts):
 *  - instance-wide: the `llm_parsing` addon's `config` JSON (admin-set, wins)
 *  - per-user: the `llm_*` keys in the per-user settings table (fallback)
 *
 * The API key is encrypted at rest (reusing apiKeyCrypto) and never returned to
 * the client in plaintext — it is masked with MASKED_VALUE, matching the
 * per-user encrypted-settings pattern in settingsService.ts.
 */

export type LlmProvider = 'local' | 'openai' | 'anthropic' | 'anthropic-compatible';

/** Fully-resolved config the clients consume. */
export interface ResolvedLlmConfig {
  provider: LlmProvider;
  model: string;
  baseUrl?: string;
  apiKey?: string;
  multimodal: boolean;
}

/** Shape of the admin instance config stored in `addons.config` (apiKey encrypted). */
export interface LlmAddonConfig {
  provider?: LlmProvider;
  model?: string;
  baseUrl?: string;
  apiKey?: string;
  multimodal?: boolean;
}

export const LLM_PROVIDERS: LlmProvider[] = ['local', 'openai', 'anthropic', 'anthropic-compatible'];
export const MASKED_VALUE = '••••••••';

/**
 * The providers that mean nothing but "an endpoint I name": 'local' is a
 * self-hosted OpenAI-compatible server (Ollama & co.), 'anthropic-compatible'
 * any server speaking the Anthropic Messages API at an address the operator
 * gives — a gateway (LiteLLM, OpenRouter) or another vendor's Anthropic-shaped
 * endpoint. Both are instance configuration (#1772): a personal write of one is
 * refused, and a per-user row naming one resolves only against an admin default
 * of the SAME provider. Every rule that asks "does this name an address" tests
 * this list, so a further endpoint provider is one entry, not a sweep.
 */
export const ENDPOINT_LLM_PROVIDERS: LlmProvider[] = ['local', 'anthropic-compatible'];

/** True when the provider names an operator-chosen endpoint (see above). */
export function isEndpointLlmProvider(value: unknown): value is LlmProvider {
  return typeof value === 'string' && (ENDPOINT_LLM_PROVIDERS as string[]).includes(value);
}

/**
 * Prepare an admin config blob for persistence: encrypt a freshly-entered apiKey,
 * and preserve the previously-stored (already-encrypted) key when the client
 * echoes back the mask sentinel (i.e. the user didn't change it).
 */
export function prepareLlmAddonConfigForWrite(
  incoming: Record<string, unknown>,
  existingStored: Record<string, unknown> | undefined,
): Record<string, unknown> {
  const out: Record<string, unknown> = { ...incoming };
  const key = incoming.apiKey;
  if (key === undefined || key === null || key === '' || key === MASKED_VALUE) {
    // Keep the existing encrypted key untouched (mask echoed or no key supplied).
    if (existingStored && 'apiKey' in existingStored) out.apiKey = existingStored.apiKey;
    else delete out.apiKey;
  } else {
    out.apiKey = maybe_encrypt_api_key(String(key)) ?? String(key);
  }
  return out;
}

/** Mask the apiKey for any client-facing response (never leak plaintext). */
export function maskLlmAddonConfig(config: Record<string, unknown>): Record<string, unknown> {
  if (config && config.apiKey) return { ...config, apiKey: MASKED_VALUE };
  return config;
}

/** Decrypt the stored apiKey for server-side use (resolver only). */
export function decryptLlmApiKey(stored: unknown): string | undefined {
  if (!stored) return undefined;
  return decrypt_api_key(stored) ?? undefined;
}
