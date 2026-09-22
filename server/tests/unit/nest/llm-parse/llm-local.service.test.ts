import { describe, it, expect, vi, beforeEach } from 'vitest';
import { HttpException } from '@nestjs/common';

// listModels/pull go through safeFetchLlm (SSRF guard: allows a local/LAN Ollama,
// blocks the cloud-metadata range). Mock it so the tests never resolve DNS; its
// (url, init) signature matches the raw fetch it replaced.
const { safeFetchLlmMock } = vi.hoisted(() => ({ safeFetchLlmMock: vi.fn() }));
vi.mock('../../../../src/utils/ssrfGuard', () => ({ safeFetchLlm: safeFetchLlmMock }));

import { LlmLocalService } from '../../../../src/nest/llm-parse/llm-local.service';
import type { LlmConfigResolver } from '../../../../src/nest/llm-parse/llm-config.resolver';

/** The service only asks the resolver for the admin's stored key. */
const resolver = (apiKey?: string) => ({ instanceApiKey: () => apiKey }) as unknown as LlmConfigResolver;
const svc = (apiKey?: string) => new LlmLocalService(resolver(apiKey));

function mockFetch(impl: any) {
  safeFetchLlmMock.mockImplementation(impl);
  return safeFetchLlmMock;
}

/** The `{ error }` envelope an HttpException carries (its .message is just 'Http Exception'). */
async function errorOf(promise: Promise<unknown>): Promise<string> {
  try {
    await promise;
  } catch (e) {
    return ((e as HttpException).getResponse() as { error?: string }).error ?? '';
  }
  throw new Error('expected the call to reject');
}

/** Read a returned NDJSON stream back into the progress lines the client parses. */
async function readLines(stream: ReadableStream<Uint8Array>): Promise<Record<string, unknown>[]> {
  const reader = stream.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
  }
  return buf.split('\n').filter(Boolean).map(l => JSON.parse(l));
}

beforeEach(() => safeFetchLlmMock.mockReset());

describe('LlmLocalService.serverRoot', () => {
  it('strips a trailing /v1 and slashes', () => {
    expect(svc().serverRoot('http://localhost:11434/v1')).toBe('http://localhost:11434');
    expect(svc().serverRoot('http://localhost:11434/v1/')).toBe('http://localhost:11434');
    expect(svc().serverRoot('http://host:1/')).toBe('http://host:1');
  });

  it('defaults to each server own port when no base URL is given', () => {
    expect(svc().serverRoot(undefined)).toBe('http://localhost:11434');
    expect(svc().serverRoot(undefined, 'lmstudio')).toBe('http://localhost:1234');
  });

  it('rejects non-http(s) and invalid URLs', () => {
    expect(() => svc().serverRoot('ftp://x')).toThrow(HttpException);
    expect(() => svc().serverRoot('not a url')).toThrow(HttpException);
    // A blank base URL is a bad value, not an absent one — 400, as before.
    expect(() => svc().serverRoot('   ', 'lmstudio')).toThrow(HttpException);
  });
});

describe('LlmLocalService.listModels', () => {
  it('returns named models from /api/tags', async () => {
    const fetchFn = mockFetch(async () => ({ ok: true, json: async () => ({ models: [{ name: 'nuextract', size: 100 }, { name: '' }] }) }));
    const out = await svc().listModels('http://localhost:11434/v1');
    expect(out.models).toEqual([{ name: 'nuextract', size: 100 }]);
    expect(fetchFn.mock.calls[0][0]).toBe('http://localhost:11434/api/tags');
  });

  it('reads LM Studio models from the native /api/v1/models, dropping embedding models', async () => {
    const fetchFn = mockFetch(async () => ({
      ok: true,
      json: async () => ({
        // v1's shape: the id is `key`, the size is reported, and an embedding model
        // says `embedding` (v0 said `id`, no size, and `embeddings`).
        models: [
          { key: 'qwen/qwen3.5-4b', type: 'llm', size_bytes: 3_400_000_000 },
          { key: 'qwen/qwen3-vl-4b', type: 'llm', size_bytes: 4_100_000_000 },
          { key: 'text-embedding-nomic-embed-text-v1.5', type: 'embedding', size_bytes: 274_290_560 },
          { key: '', type: 'llm' },
        ],
      }),
    }));
    const out = await svc().listModels('http://localhost:1234/v1', 'lmstudio');
    expect(out.models).toEqual([
      { name: 'qwen/qwen3.5-4b', size: 3_400_000_000 },
      { name: 'qwen/qwen3-vl-4b', size: 4_100_000_000 },
    ]);
    expect(fetchFn.mock.calls[0][0]).toBe('http://localhost:1234/api/v1/models');
  });

  it('sends the admin stored key as a bearer token, and nothing when there is none', async () => {
    const fetchFn = mockFetch(async () => ({ ok: true, json: async () => ({ models: [] }) }));
    await svc('lms-token').listModels('http://localhost:1234/v1', 'lmstudio');
    expect(fetchFn.mock.calls[0][1].headers).toMatchObject({ authorization: 'Bearer lms-token' });

    await svc().listModels('http://localhost:1234/v1', 'lmstudio');
    expect(fetchFn.mock.calls[1][1].headers).toEqual({});
  });

  it('says a refused request wants a key, rather than reporting a bare status', async () => {
    mockFetch(async () => ({ ok: false, status: 401 }));
    // A v1 LM Studio can be configured to require an API token; "error (401)" would
    // send the admin looking at the model list instead of the key field.
    expect(await errorOf(svc().listModels('http://localhost:1234/v1', 'lmstudio'))).toMatch(/wants an API key/);
  });

  it('falls back to Ollama for an unknown or absent provider', async () => {
    const fetchFn = mockFetch(async () => ({ ok: true, json: async () => ({ models: [] }) }));
    await svc().listModels('http://localhost:11434/v1', 'something-else');
    expect(fetchFn.mock.calls[0][0]).toBe('http://localhost:11434/api/tags');
  });

  it('502s when the server is unreachable', async () => {
    // Reject only the one call listModels makes (mockImplementationOnce): vitest
    // probes the mock a second time and a persistent rejection there would surface
    // as an unhandled rejection and fail the test even though listModels catches
    // the real one and maps it to a 502.
    safeFetchLlmMock.mockImplementationOnce(() => Promise.reject(new Error('ECONNREFUSED')));
    await expect(svc().listModels('http://localhost:11434')).rejects.toThrow(HttpException);
  });
});

describe('LlmLocalService.pull', () => {
  it('requires a model', async () => {
    await expect(svc().pull('http://localhost:11434', '')).rejects.toThrow(HttpException);
    await expect(svc().pull('http://localhost:1234/v1', '  ', 'lmstudio')).rejects.toThrow(HttpException);
  });

  it('posts to /api/pull and returns the stream body', async () => {
    const body = {} as ReadableStream<Uint8Array>;
    const fetchFn = mockFetch(async () => ({ ok: true, body }));
    const out = await svc().pull('http://localhost:11434/v1', 'nuextract');
    expect(out).toBe(body);
    expect(fetchFn.mock.calls[0][0]).toBe('http://localhost:11434/api/pull');
    const init = fetchFn.mock.calls[0][1];
    expect(JSON.parse(init.body)).toEqual({ model: 'nuextract', stream: true });
  });

  it('starts an LM Studio download on the v1 endpoint', async () => {
    const fetchFn = mockFetch(async () => ({ ok: true, json: async () => ({ status: 'already_downloaded' }) }));
    const stream = await svc('lms-token').pull('http://localhost:1234/v1', '  qwen/qwen3.5-4b  ', 'lmstudio');

    expect(fetchFn.mock.calls[0][0]).toBe('http://localhost:1234/api/v1/models/download');
    const init = fetchFn.mock.calls[0][1];
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ model: 'qwen/qwen3.5-4b' });
    expect(init.headers).toMatchObject({ authorization: 'Bearer lms-token' });
    // A model already on disk is a finished pull, not an error: the panel selects it.
    expect(await readLines(stream)).toEqual([{ status: 'already downloaded' }]);
  });

  it('explains a download LM Studio would not start, instead of a bare 502', async () => {
    mockFetch(async () => ({ ok: false, status: 404 }));
    expect(await errorOf(svc().pull('http://localhost:1234/v1', 'not-a-catalog-id', 'lmstudio'))).toMatch(
      /LM Studio model catalog/,
    );
  });

  it('tells a refused download from an unknown one, and both from an unreachable server', async () => {
    // Same failure to the caller, three different things for the admin to do.
    mockFetch(async () => ({ ok: false, status: 401 }));
    expect(await errorOf(svc().pull('http://localhost:1234/v1', 'qwen/qwen3.5-4b', 'lmstudio'))).toMatch(
      /wants an API key/,
    );

    safeFetchLlmMock.mockImplementationOnce(() => Promise.reject(new Error('ECONNREFUSED')));
    expect(await errorOf(svc().pull('http://localhost:1234/v1', 'qwen/qwen3.5-4b', 'lmstudio'))).toMatch(
      /Could not reach/,
    );
  });
});
