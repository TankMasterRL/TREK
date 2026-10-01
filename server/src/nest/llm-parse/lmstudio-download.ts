/**
 * LM Studio model download, bridged to the progress stream the admin panel already reads.
 *
 * The two servers hand over the same information in opposite directions. Ollama pushes:
 * `POST /api/pull` answers with a live NDJSON stream and keeps writing until the model
 * is on disk. LM Studio's native v1 API hands back a job — `POST /api/v1/models/download`
 * answers `{ job_id, status, total_size_bytes }` and returns, and
 * `GET /api/v1/models/download/status/:job_id` reports how far it has got.
 *
 * So the polling lives here, and the difference stops here: this produces the same
 * NDJSON stream of `{ status, total?, completed? }` lines Ollama emits, which means the
 * controller, the admin API client and the progress bar in both shells are untouched by
 * LM Studio having a different kind of API. One translation in one place beats a second
 * progress protocol carried all the way to the browser.
 */

import { safeFetchLlm } from '../../utils/ssrfGuard';

/** LM Studio's report on one download job (both the start call and the status poll). */
interface DownloadJob {
  job_id?: string;
  status?: 'downloading' | 'paused' | 'completed' | 'failed' | 'already_downloaded';
  total_size_bytes?: number;
  downloaded_bytes?: number;
}

/** How often the job is asked for progress. Matches the cadence Ollama pushes at. */
const POLL_INTERVAL_MS = 1000;

/**
 * Give up after this many status reads in a row that did not answer. A download
 * survives a blip; a server that has stopped answering altogether is an outage, and
 * the admin should be told rather than left watching a bar that will never move.
 */
const MAX_CONSECUTIVE_FAILURES = 5;

/**
 * Stop following a job after this long. Not a limit on how big a model may be — it is
 * the end of a stream nobody is going to see finish, e.g. a job left `paused` in the
 * LM Studio app. The download itself keeps running there; only this progress view ends.
 */
const MAX_FOLLOW_MS = 2 * 60 * 60 * 1000;

/** Request timeout for one job call, matching the model-list read next door. */
const REQUEST_TIMEOUT_MS = 10_000;

/** Ollama's line shape, which the client's progress handler is written against. */
interface ProgressLine {
  status?: string;
  total?: number;
  completed?: number;
  error?: string;
}

const auth = (apiKey?: string) => (apiKey ? { authorization: `Bearer ${apiKey}` } : {});

/**
 * Thrown when the download never started, carrying the upstream status (0 = the server
 * could not be reached at all). The caller turns it into the right message: a 401 wants
 * an API token, anything else is usually a model id LM Studio's catalog does not know —
 * two very different things to tell an admin.
 */
export class LmStudioDownloadStartError extends Error {
  constructor(readonly status: number) {
    super(`LM Studio would not start the download (${status || 'unreachable'})`);
    this.name = 'LmStudioDownloadStartError';
  }
}

/** Read a job report, or null when the call failed or answered something unreadable. */
async function readJob(url: string, init: RequestInit): Promise<DownloadJob | null> {
  try {
    const res = await safeFetchLlm(url, { ...init, signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
    if (!res.ok) return null;
    const data = (await res.json()) as DownloadJob;
    return data && typeof data === 'object' ? data : null;
  } catch {
    return null;
  }
}

/** One job report → the progress line for it, or null while it is still running. */
function terminalLine(job: DownloadJob): ProgressLine | null {
  if (job.status === 'completed') {
    return { status: 'success', total: job.total_size_bytes, completed: job.total_size_bytes };
  }
  // The model is already on disk. Not an error — the panel selects it and refreshes,
  // which is exactly what a finished download does.
  if (job.status === 'already_downloaded') return { status: 'already downloaded' };
  if (job.status === 'failed') return { error: 'LM Studio reported the download as failed' };
  return null;
}

/**
 * Start a download on LM Studio and follow it to the end.
 *
 * Returns the NDJSON progress stream described at the top of this file. Errors that
 * happen before anything is written are thrown, so the caller can still answer with the
 * normal `{ error }` envelope; anything after that is reported as an `{ error }` LINE,
 * because by then the client is reading a 200 body.
 */
export async function downloadLmStudioModel(
  root: string,
  model: string,
  apiKey?: string,
): Promise<ReadableStream<Uint8Array>> {
  const started = await startDownload(root, model, apiKey);

  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const emit = (line: ProgressLine) => controller.enqueue(encoder.encode(`${JSON.stringify(line)}\n`));
      try {
        await followJob(root, started, apiKey, emit);
      } catch {
        // The consumer hung up mid-poll; there is nobody left to report to.
      } finally {
        controller.close();
      }
    },
  });
}

/** POST the download, or throw with why it did not start. */
async function startDownload(root: string, model: string, apiKey?: string): Promise<DownloadJob> {
  let res: Response;
  try {
    res = await safeFetchLlm(`${root}/api/v1/models/download`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...auth(apiKey) },
      body: JSON.stringify({ model }),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch {
    throw new LmStudioDownloadStartError(0);
  }
  if (!res.ok) throw new LmStudioDownloadStartError(res.status);
  const job = (await res.json().catch(() => null)) as DownloadJob | null;
  if (!job || typeof job !== 'object') throw new LmStudioDownloadStartError(res.status);
  return job;
}

/**
 * Emit the progress lines for one started job.
 *
 * Always ends on a line that says HOW it ended — never on silence. The client reads a
 * stream that simply stopped as a finished pull, so a view that ran out of time or lost
 * the server has to say so or a download that never landed is reported as a success.
 */
async function followJob(
  root: string,
  started: DownloadJob,
  apiKey: string | undefined,
  emit: (line: ProgressLine) => void,
): Promise<void> {
  const settled = terminalLine(started);
  if (settled || !started.job_id) {
    // Nothing to follow: either it finished on the spot, or LM Studio accepted the
    // request without naming a job, which there is no way to report progress for.
    emit(settled ?? { status: 'success', total: started.total_size_bytes, completed: started.total_size_bytes });
    return;
  }

  emit({ status: 'starting download', total: started.total_size_bytes, completed: 0 });
  const statusUrl = `${root}/api/v1/models/download/status/${encodeURIComponent(started.job_id)}`;
  const deadline = Date.now() + MAX_FOLLOW_MS;
  let last: ProgressLine = { error: 'Gave up following the download — it may still be running in LM Studio' };
  let failures = 0;

  while (Date.now() < deadline) {
    await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL_MS));
    const job = await readJob(statusUrl, { headers: auth(apiKey) });
    if (!job) {
      if (++failures >= MAX_CONSECUTIVE_FAILURES) {
        last = { error: 'Lost contact with LM Studio while the model was downloading' };
        break;
      }
      continue;
    }
    failures = 0;

    const done = terminalLine(job);
    if (done) {
      last = done;
      break;
    }
    emit({
      status: job.status === 'paused' ? 'paused' : 'downloading',
      total: job.total_size_bytes,
      completed: job.downloaded_bytes,
    });
  }
  emit(last);
}
