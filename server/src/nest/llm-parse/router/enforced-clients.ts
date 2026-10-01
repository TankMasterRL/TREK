/**
 * Which enforced-extraction client answers for which self-hosted provider.
 *
 * Separate from ./enforced.ts so the shared plumbing stays a leaf the clients can
 * import without a cycle, and separate from the router so the router never grows a
 * per-provider `if`. Adding a third local server is one entry here plus its client.
 */

import type { LlmProvider } from '../llm-config';
import type { EnforcedExtractor } from './enforced';
import { extractEnforced } from './ollama-format.client';
import { extractEnforcedLmStudio } from './lmstudio-format.client';

const EXTRACTORS: Partial<Record<LlmProvider, EnforcedExtractor>> = {
  local: extractEnforced,
  lmstudio: extractEnforcedLmStudio,
};

/**
 * The extractor for a provider. Ollama is the fallback for anything unknown: it is
 * the provider the router was built against, and a config that reached the router at
 * all has already been through the self-hosted gate in llm-config.resolver.ts.
 */
export function enforcedExtractorFor(provider: LlmProvider | undefined): EnforcedExtractor {
  return (provider && EXTRACTORS[provider]) || extractEnforced;
}
