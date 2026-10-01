import type { LlmExtractionClient } from './llm-provider.interface';
import type { ResolvedLlmConfig } from './llm-config';
import { OpenAiCompatibleClient } from './clients/openai-compatible.client';
import { AnthropicClient } from './clients/anthropic.client';

/**
 * Pick the provider client for a resolved config.
 *  - 'anthropic'                      → Anthropic Messages API client
 *  - 'openai' | 'local' | 'lmstudio'  → OpenAI-compatible client (cloud or local base URL)
 *
 * The two self-hosted providers normally never get here: llm-parse.service.ts routes
 * them through the schema-enforcing extraction router instead. This stays their client
 * for the one case the router cannot take — a document that produced no text at all —
 * and because both servers do speak the OpenAI-compatible shape.
 */
export function createLlmClient(config: ResolvedLlmConfig): LlmExtractionClient {
  switch (config.provider) {
    case 'anthropic':
      return new AnthropicClient();
    case 'openai':
    case 'local':
    case 'lmstudio':
      return new OpenAiCompatibleClient();
    // TODO(nuextract): add a NuExtract template adapter here (local vision model
    // with its own template-fill API) once the OpenAI-compatible path proves
    // insufficient for small local models — see the design seam in the plan.
    default:
      return new OpenAiCompatibleClient();
  }
}
