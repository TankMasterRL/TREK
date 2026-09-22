import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// The download goes through safeFetchLlm, like every other outbound LLM call.
const { safeFetchLlmMock } = vi.hoisted(() => ({ safeFetchLlmMock: vi.fn() }));
vi.mock('../../../../src/utils/ssrfGuard', () => ({ safeFetchLlm: safeFetchLlmMock }));

import { downloadLmStudioModel, LmStudioDownloadStartError } from '../../../../src/nest/llm-parse/lmstudio-download';

const ROOT = 'http://localhost:1234';

/** A `{ ok, json }` stand-in for one upstream answer. */
const reply = (body: unknown) => ({ ok: true, json: async () => body });

/**
 * Answer the start call with `start`, then each status poll with the next `polls` entry
 * (the last one repeats). `null` stands for a call that fails outright.
 */
function upstream(start: unknown, polls: unknown[] = []) {
  let seen = 0;
  safeFetchLlmMock.mockImplementation(async (url: string) => {
    if (url.endsWith('/api/v1/models/download')) {
      if (start === null) throw new Error('ECONNREFUSED');
      return reply(start);
    }
    const next = polls[Math.min(seen++, polls.length - 1)];
    if (next === null) throw new Error('ECONNREFUSED');
    return reply(next);
  });
}

/** Drain the NDJSON stream into the progress lines the admin panel parses. */
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

beforeEach(() => {
  safeFetchLlmMock.mockReset();
  // The poll waits a second between reads; fake timers keep the suite instant. The
  // stream is drained with runAllTimersAsync so each await actually resolves.
  vi.useFakeTimers();
});
afterEach(() => vi.useRealTimers());

/** Start a download and drain it while the fake clock runs the polls. */
async function run(): Promise<Record<string, unknown>[]> {
  const stream = await downloadLmStudioModel(ROOT, 'qwen/qwen3.5-4b');
  const lines = readLines(stream);
  await vi.runAllTimersAsync();
  return lines;
}

describe('downloadLmStudioModel', () => {
  it('follows a job to completion as Ollama-shaped progress lines', async () => {
    upstream(
      { job_id: 'job_1', status: 'downloading', total_size_bytes: 1000 },
      [
        { status: 'downloading', total_size_bytes: 1000, downloaded_bytes: 400 },
        { status: 'downloading', total_size_bytes: 1000, downloaded_bytes: 900 },
        { status: 'completed', total_size_bytes: 1000, downloaded_bytes: 1000 },
      ],
    );

    // `{ status, total, completed }` is Ollama's line shape, which is the whole point:
    // the controller, the API client and the progress bar in both shells never learn
    // that LM Studio reports a job rather than streaming.
    expect(await run()).toEqual([
      { status: 'starting download', total: 1000, completed: 0 },
      { status: 'downloading', total: 1000, completed: 400 },
      { status: 'downloading', total: 1000, completed: 900 },
      { status: 'success', total: 1000, completed: 1000 },
    ]);
  });

  it('polls the job that the start call named', async () => {
    upstream({ job_id: 'job_493c7c9ded', status: 'downloading' }, [{ status: 'completed' }]);
    await run();
    expect(safeFetchLlmMock.mock.calls[1][0]).toBe(`${ROOT}/api/v1/models/download/status/job_493c7c9ded`);
  });

  it('reports a model already on disk as a finished pull, not an error', async () => {
    upstream({ status: 'already_downloaded' });
    expect(await run()).toEqual([{ status: 'already downloaded' }]);
    // Nothing to follow, so no status call goes out at all.
    expect(safeFetchLlmMock).toHaveBeenCalledTimes(1);
  });

  it('passes a paused job through rather than calling it finished', async () => {
    upstream(
      { job_id: 'job_1', status: 'downloading', total_size_bytes: 500 },
      [{ status: 'paused', total_size_bytes: 500, downloaded_bytes: 120 }, { status: 'completed', total_size_bytes: 500 }],
    );
    const lines = await run();
    expect(lines[1]).toEqual({ status: 'paused', total: 500, completed: 120 });
  });

  it('ends a failed download on an error line', async () => {
    upstream({ job_id: 'job_1', status: 'downloading' }, [{ status: 'failed' }]);
    // The client throws on a line carrying `error`, so the admin sees a failed pull
    // instead of a success toast for a model that never landed.
    expect((await run()).at(-1)).toEqual({ error: 'LM Studio reported the download as failed' });
  });

  it('gives up with an error line once the server stops answering', async () => {
    upstream({ job_id: 'job_1', status: 'downloading' }, [null]);
    const lines = await run();
    expect(lines.at(-1)).toEqual({ error: 'Lost contact with LM Studio while the model was downloading' });
    // A blip is survivable, so it takes several failures in a row to stop.
    expect(safeFetchLlmMock.mock.calls.length).toBeGreaterThan(2);
  });

  it('rides out a single failed poll', async () => {
    upstream(
      { job_id: 'job_1', status: 'downloading', total_size_bytes: 10 },
      [null, { status: 'downloading', total_size_bytes: 10, downloaded_bytes: 5 }, { status: 'completed', total_size_bytes: 10 }],
    );
    const lines = await run();
    expect(lines.at(-1)).toEqual({ status: 'success', total: 10, completed: 10 });
  });

  it('throws when the download never starts, so the caller can still answer JSON', async () => {
    upstream(null);
    // Nothing has been written yet, so this becomes the normal `{ error }` envelope
    // rather than an error line on a 200 body. Status 0 = the server never answered.
    await expect(downloadLmStudioModel(ROOT, 'nope')).rejects.toMatchObject({ status: 0 });
  });

  it('carries the refusal status, so a missing token is not reported as a bad model id', async () => {
    safeFetchLlmMock.mockImplementation(async () => ({ ok: false, status: 401 }));
    const err = await downloadLmStudioModel(ROOT, 'qwen/qwen3.5-4b').catch(e => e);
    expect(err).toBeInstanceOf(LmStudioDownloadStartError);
    expect(err.status).toBe(401);
  });
});
