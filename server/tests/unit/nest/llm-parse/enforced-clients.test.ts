import { describe, it, expect } from 'vitest';
import { enforcedExtractorFor } from '../../../../src/nest/llm-parse/router/enforced-clients';
import { extractEnforced } from '../../../../src/nest/llm-parse/router/ollama-format.client';
import { extractEnforcedLmStudio } from '../../../../src/nest/llm-parse/router/lmstudio-format.client';
import { SELF_HOSTED_LLM_PROVIDERS } from '../../../../src/nest/llm-parse/llm-config';

describe('enforcedExtractorFor', () => {
  it('routes each self-hosted provider to its own client', () => {
    expect(enforcedExtractorFor('local')).toBe(extractEnforced);
    expect(enforcedExtractorFor('lmstudio')).toBe(extractEnforcedLmStudio);
  });

  it('falls back to Ollama for an unknown or absent provider', () => {
    expect(enforcedExtractorFor(undefined)).toBe(extractEnforced);
    expect(enforcedExtractorFor('openai')).toBe(extractEnforced);
  });

  it('has a client for every self-hosted provider', () => {
    // The router is only ever reached by a self-hosted provider, so a new one
    // added to the list without a client here would silently be sent to Ollama's
    // native API — a 404 from a server that does not have it.
    const distinct = new Set(SELF_HOSTED_LLM_PROVIDERS.map(p => enforcedExtractorFor(p)));
    expect(distinct.size).toBe(SELF_HOSTED_LLM_PROVIDERS.length);
  });
});
