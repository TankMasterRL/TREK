import { describe, it, expect } from 'vitest';
import { toNativeBase, toOpenAiBase } from '../../../../src/nest/llm-parse/router/enforced';

// The base URL an admin configures is one field for every provider, so both
// helpers have to cope with whichever form they were handed. Both are used to
// build a URL the SSRF guard then checks, so a doubled or missing segment is a
// request to the wrong place, not a cosmetic difference.
describe('toNativeBase', () => {
  it('strips a /v1 suffix and trailing slashes', () => {
    expect(toNativeBase('http://ollama:11434/v1')).toBe('http://ollama:11434');
    expect(toNativeBase('http://ollama:11434/v1/')).toBe('http://ollama:11434');
    expect(toNativeBase('http://ollama:11434/')).toBe('http://ollama:11434');
    expect(toNativeBase('http://ollama:11434')).toBe('http://ollama:11434');
  });

  it('leaves a path that merely ends in something else alone', () => {
    expect(toNativeBase('http://host/proxy/ollama')).toBe('http://host/proxy/ollama');
  });
});

describe('toOpenAiBase', () => {
  it('normalizes to exactly one /v1, however the admin spelled the base URL', () => {
    expect(toOpenAiBase('http://lmstudio:1234')).toBe('http://lmstudio:1234/v1');
    expect(toOpenAiBase('http://lmstudio:1234/')).toBe('http://lmstudio:1234/v1');
    expect(toOpenAiBase('http://lmstudio:1234/v1')).toBe('http://lmstudio:1234/v1');
    expect(toOpenAiBase('http://lmstudio:1234/v1/')).toBe('http://lmstudio:1234/v1');
  });
});
