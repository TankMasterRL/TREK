import { describe, it, expect, vi, beforeEach } from 'vitest';

// The client goes through safeFetchLlm (SSRF guard: blocks the cloud-metadata
// range, allows a local/LAN model server). Mock it so the tests never do a real
// DNS lookup; its (url, init) signature matches the raw fetch it wraps, so the
// recorded-call assertions are unchanged.
const { safeFetchLlmMock } = vi.hoisted(() => ({ safeFetchLlmMock: vi.fn() }));
vi.mock('../../../../src/utils/ssrfGuard', () => ({ safeFetchLlm: safeFetchLlmMock }));

import { extractEnforcedLmStudio } from '../../../../src/nest/llm-parse/router/lmstudio-format.client';

function mockFetch(impl: (url: string, init: RequestInit) => Promise<Response> | Response) {
  safeFetchLlmMock.mockImplementation(impl as unknown as typeof fetch);
  return safeFetchLlmMock;
}

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return { ok, status, json: async () => body, text: async () => JSON.stringify(body) } as unknown as Response;
}

/** LM Studio's envelope: the constrained JSON arrives as a string in the message content. */
function completion(content: string) {
  return { choices: [{ message: { content } }] };
}

const INPUT = {
  baseUrl: 'http://lmstudio:1234/v1',
  model: 'qwen3-8b',
  system: 'sys',
  user: 'doc',
  schema: { type: 'object' as const },
};

beforeEach(() => safeFetchLlmMock.mockReset());

describe('extractEnforcedLmStudio', () => {
  it('posts to /v1/chat/completions with a strict json_schema response format', async () => {
    const fetchFn = mockFetch(() => jsonResponse(completion('{"name":"Hotel"}')));
    const out = await extractEnforcedLmStudio(INPUT);
    expect(out).toEqual({ name: 'Hotel' });

    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe('http://lmstudio:1234/v1/chat/completions');
    const body = JSON.parse((init as RequestInit).body as string);
    // `strict` is what makes LM Studio constrain sampling rather than merely ask.
    expect(body.response_format).toEqual({
      type: 'json_schema',
      json_schema: { name: 'extraction', strict: true, schema: { type: 'object' } },
    });
    expect(body.model).toBe('qwen3-8b');
    expect(body.temperature).toBe(0);
    expect(body.stream).toBe(false);
    expect(body.messages).toEqual([
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'doc' },
    ]);
    expect((init as RequestInit).headers).not.toHaveProperty('authorization');
  });

  it('appends /v1 when the configured base URL omits it', async () => {
    const fetchFn = mockFetch(() => jsonResponse(completion('{}')));
    await extractEnforcedLmStudio({ ...INPUT, baseUrl: 'http://lmstudio:1234' });
    expect(fetchFn.mock.calls[0][0]).toBe('http://lmstudio:1234/v1/chat/completions');
  });

  it('sends a bearer header only when an apiKey is given, and honours numPredict', async () => {
    const fetchFn = mockFetch(() => jsonResponse(completion('{}')));
    await extractEnforcedLmStudio({ ...INPUT, apiKey: 'lms-123', numPredict: 900 });
    const init = fetchFn.mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>).authorization).toBe('Bearer lms-123');
    expect(JSON.parse(init.body as string).max_tokens).toBe(900);
  });

  it('strips a ```json code fence before parsing', async () => {
    mockFetch(() => jsonResponse(completion('```json\n{"a":1}\n```')));
    expect(await extractEnforcedLmStudio(INPUT)).toEqual({ a: 1 });
  });

  it('drops a leading reasoning block a hybrid model printed ahead of the answer', async () => {
    // LM Studio has no `think: false` switch the way Ollama does, so a reasoning
    // model can print its chain of thought first. Without this the whole answer
    // parses to null and a good extraction is lost.
    mockFetch(() => jsonResponse(completion('<think>the total is in euros</think>{"a":1}')));
    expect(await extractEnforcedLmStudio(INPUT)).toEqual({ a: 1 });

    mockFetch(() => jsonResponse(completion('\n  <Reasoning>weighing it up</Reasoning>\n```json\n{"a":2}\n```')));
    expect(await extractEnforcedLmStudio(INPUT)).toEqual({ a: 2 });
  });

  it('leaves content alone when the reasoning block never closes', async () => {
    // Half a tag is not a block to cut at — dropping to the end of the string
    // would throw away an answer that came after it.
    mockFetch(() => jsonResponse(completion('<think>{"a":1}')));
    expect(await extractEnforcedLmStudio(INPUT)).toBeNull();
    mockFetch(() => jsonResponse(completion('{"think":"<think>"}')));
    expect(await extractEnforcedLmStudio(INPUT)).toEqual({ think: '<think>' });
  });

  it('returns null when the content parses to a non-object', async () => {
    mockFetch(() => jsonResponse(completion('"just a string"')));
    expect(await extractEnforcedLmStudio(INPUT)).toBeNull();
  });

  it('returns null for unparseable content and for a response with no choices', async () => {
    mockFetch(() => jsonResponse(completion('not json at all')));
    expect(await extractEnforcedLmStudio(INPUT)).toBeNull();
    mockFetch(() => jsonResponse({}));
    expect(await extractEnforcedLmStudio(INPUT)).toBeNull();
  });

  it('throws naming LM Studio and the status when it responds non-ok', async () => {
    mockFetch(() => jsonResponse({ error: 'model not loaded' }, false, 404));
    // The message has to name the server the admin configured — "the LLM failed"
    // sends them looking in the wrong place.
    await expect(extractEnforcedLmStudio(INPUT)).rejects.toThrow(/LM Studio \/v1\/chat\/completions failed \(404\)/);
  });
});
