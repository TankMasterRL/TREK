import { Injectable, HttpException } from '@nestjs/common';
import { safeFetchLlm } from '../../utils/ssrfGuard';
import { isSelfHostedLlmProvider, type LlmProvider } from './llm-config';
import { LlmConfigResolver } from './llm-config.resolver';
import { downloadLmStudioModel, LmStudioDownloadStartError } from './lmstudio-download';

/** What the addon UI needs about one model installed on a self-hosted server. */
export interface LocalModel {
  name: string;
  /** Bytes on disk, as the server reports them. */
  size: number;
}

/** Ollama's default when the admin has not named an endpoint yet. */
const DEFAULT_ROOT = 'http://localhost:11434';
/** LM Studio's default — its own port, and a different one. */
const DEFAULT_LMSTUDIO_ROOT = 'http://localhost:1234';

/**
 * Admin helpers for managing a local LLM server: list the models it already has, and
 * download a new one.
 *
 * Both servers keep this *management* API separate from the inference path the
 * extraction router uses, and both hang it off the server root rather than the `/v1`
 * OpenAI-compatible prefix an admin normally configures, so the root is derived by
 * stripping it:
 *
 *  - Ollama: `/api/tags` and `/api/pull`.
 *  - LM Studio: its **native v1 REST API** — `/api/v1/models` and
 *    `/api/v1/models/download`. That API (LM Studio 0.4.0) replaces the `/api/v0/*`
 *    one, which LM Studio's docs now mark as superseded, and it is what adds a
 *    download endpoint at all: v0 had none, so TREK used to answer a pull for LM
 *    Studio with "do it in the app yourself".
 *
 * Admin-only (guarded at the controller); the base URL is admin-supplied. Requests go
 * through safeFetchLlm, which still allows a localhost/LAN server but blocks the
 * link-local / cloud-metadata range. A configured API key is sent as a bearer token:
 * LM Studio's v1 server can be set to require one, and an Ollama behind a reverse
 * proxy may too — the same key, sent the same way as on the extraction path.
 */
@Injectable()
export class LlmLocalService {
  constructor(private readonly config: LlmConfigResolver) {}

  /** Derive the management-API root from a configured base URL (strip a trailing /v1). */
  serverRoot(baseUrl: string | undefined, provider?: string): string {
    // Only an ABSENT base URL falls back. A blank one is a caller sending something
    // wrong, and it answered 400 before LM Studio existed — `|| fallback` here would
    // quietly turn that into a request to localhost instead.
    const fallback = provider === 'lmstudio' ? DEFAULT_LMSTUDIO_ROOT : DEFAULT_ROOT;
    const raw = (baseUrl ?? fallback).trim();
    let url: URL;
    try {
      url = new URL(raw);
    } catch {
      throw new HttpException({ error: 'Invalid base URL' }, 400);
    }
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      throw new HttpException({ error: 'Base URL must be http(s)' }, 400);
    }
    // Trailing slashes are scanned off instead of `/\/+$/`-replaced: that pattern
    // re-walks the slash run from every start position of the admin-supplied URL.
    let end = raw.length;
    while (end > 0 && raw.charCodeAt(end - 1) === 47 /* '/' */) end--;
    return raw.slice(0, end).replace(/\/v1$/, '');
  }

  /**
   * Which server a request is for. Anything that is not a known self-hosted provider
   * means Ollama — that is what the route answered for before LM Studio existed, and
   * the client omits the parameter on an install that predates it.
   */
  private resolveProvider(provider: string | undefined): LlmProvider {
    return isSelfHostedLlmProvider(provider) ? provider : 'local';
  }

  /** Bearer header for the admin-stored key, or nothing when no key is configured. */
  private authHeaders(): Record<string, string> {
    const key = this.config.instanceApiKey();
    return key ? { authorization: `Bearer ${key}` } : {};
  }

  /** List models already installed on the local server. */
  async listModels(baseUrl: string | undefined, provider?: string): Promise<{ models: LocalModel[] }> {
    const kind = this.resolveProvider(provider);
    const root = this.serverRoot(baseUrl, kind);
    const path = kind === 'lmstudio' ? '/api/v1/models' : '/api/tags';
    let res: Response;
    try {
      res = await safeFetchLlm(`${root}${path}`, {
        headers: this.authHeaders(),
        signal: AbortSignal.timeout(10_000),
      });
    } catch {
      throw new HttpException({ error: `Could not reach local LLM server at ${root}` }, 502);
    }
    if (!res.ok) throw new HttpException({ error: unreachableReason(res.status, root) }, 502);
    const data = (await res.json()) as LocalModelsResponse;
    return { models: kind === 'lmstudio' ? lmStudioModels(data) : ollamaModels(data) };
  }

  /**
   * Start a model download and return NDJSON progress lines, which the controller pipes
   * straight to the client's progress bar.
   *
   * Ollama's `/api/pull` IS that stream, so its body is handed over as it arrives. LM
   * Studio's v1 API starts a job and reports on it when asked, so lmstudio-download.ts
   * follows the job and writes the same lines — one shape reaches the browser either way.
   */
  async pull(baseUrl: string | undefined, model: string, provider?: string): Promise<ReadableStream<Uint8Array>> {
    if (!model?.trim()) throw new HttpException({ error: 'model is required' }, 400);
    const kind = this.resolveProvider(provider);
    const root = this.serverRoot(baseUrl, kind);
    if (kind === 'lmstudio') return this.pullLmStudio(root, model.trim());

    let res: Response;
    try {
      res = await safeFetchLlm(`${root}/api/pull`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', ...this.authHeaders() },
        body: JSON.stringify({ model: model.trim(), stream: true }),
      });
    } catch {
      throw new HttpException({ error: `Could not reach local LLM server at ${root}` }, 502);
    }
    if (!res.ok || !res.body) throw new HttpException({ error: `Pull failed (${res.status})` }, 502);
    return res.body;
  }

  /**
   * LM Studio downloads by catalog identifier (`qwen/qwen3.5-4b`) or Hugging Face link.
   * A model id that names neither is refused by LM Studio rather than downloaded, so the
   * failure is reported as it is — the admin can also still add the model in the app.
   */
  private async pullLmStudio(root: string, model: string): Promise<ReadableStream<Uint8Array>> {
    try {
      return await downloadLmStudioModel(root, model, this.config.instanceApiKey());
    } catch (e) {
      const status = e instanceof LmStudioDownloadStartError ? e.status : 0;
      if (!status) throw new HttpException({ error: `Could not reach local LLM server at ${root}` }, 502);
      if (status === 401 || status === 403) throw new HttpException({ error: unreachableReason(status, root) }, 502);
      throw new HttpException(
        {
          error: `LM Studio would not start a download for "${model}" (${status}). Check the id against the LM Studio model catalog, or add it in the app and press Refresh.`,
        },
        502,
      );
    }
  }
}

/** Why a management call came back non-ok, in terms the admin can act on. */
function unreachableReason(status: number, root: string): string {
  // LM Studio's v1 server can be configured to require an API token, and Ollama is
  // often put behind an authenticating proxy. "Local LLM server error (401)" sends an
  // admin looking at the model list; the key field is what actually needs filling in.
  if (status === 401 || status === 403) {
    return `Local LLM server at ${root} refused the request (${status}) — it wants an API key. Enter one above and save.`;
  }
  return `Local LLM server error (${status})`;
}

/** Both servers' list shapes — Ollama answers `{models:[{name,size}]}`, LM Studio's v1 `{models:[{key,…}]}`. */
interface LocalModelsResponse {
  models?: { name?: string; size?: number; key?: string; type?: string; size_bytes?: number }[];
}

function ollamaModels(data: LocalModelsResponse): LocalModel[] {
  return (data.models ?? []).map(m => ({ name: m.name ?? '', size: m.size ?? 0 })).filter(m => m.name);
}

/**
 * LM Studio lists every downloaded model, embedding models included. An embedding model
 * cannot answer a chat completion, so offering one would only produce a confusing failure
 * at import time — `type` says which is which ('llm' | 'embedding'); an unfamiliar type is
 * kept rather than hidden.
 *
 * Both fields moved with the v1 API: the id is `key` (v0 called it `id`) and the type of an
 * embedding model is `embedding`, singular (v0 said `embeddings`). v0's separate `vlm` type
 * is gone too — a vision model is an `llm` that reports `capabilities.vision`. The one real
 * gain for this list is `size_bytes`: v0 reported no size at all, so every LM Studio model
 * used to show as 0 bytes.
 */
function lmStudioModels(data: LocalModelsResponse): LocalModel[] {
  return (data.models ?? [])
    .filter(m => m.type !== 'embedding')
    .map(m => ({ name: m.key ?? '', size: m.size_bytes ?? 0 }))
    .filter(m => m.name);
}
