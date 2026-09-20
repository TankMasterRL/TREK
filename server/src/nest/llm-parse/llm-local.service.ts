import { Injectable, HttpException } from '@nestjs/common';
import { safeFetchLlm } from '../../utils/ssrfGuard';
import { isSelfHostedLlmProvider, type LlmProvider } from './llm-config';

/** What the addon UI needs about one model installed on a self-hosted server. */
export interface LocalModel {
  name: string;
  /** Bytes on disk. Ollama reports it; LM Studio's model list does not, so 0 there. */
  size: number;
}

/** Ollama's default when the admin has not named an endpoint yet. */
const DEFAULT_ROOT = 'http://localhost:11434';
/** LM Studio's default — its own port, and a different one. */
const DEFAULT_LMSTUDIO_ROOT = 'http://localhost:1234';

/**
 * Admin helpers for managing a local LLM server: list the models it already has,
 * and (Ollama only) pull a new one.
 *
 * Both servers keep this *management* API separate from the inference path the
 * extraction router uses: Ollama's `/api/tags` + `/api/pull` and LM Studio's
 * `/api/v0/models` live at the server root, not under the `/v1` OpenAI-compatible
 * prefix an admin normally configures, so the root is derived by stripping it.
 *
 * Admin-only (guarded at the controller); the base URL is admin-supplied. Requests
 * go through safeFetchLlm, which still allows a localhost/LAN server but blocks the
 * link-local / cloud-metadata range.
 */
@Injectable()
export class LlmLocalService {
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

  /** List models already installed on the local server. */
  async listModels(baseUrl: string | undefined, provider?: string): Promise<{ models: LocalModel[] }> {
    const kind = this.resolveProvider(provider);
    const root = this.serverRoot(baseUrl, kind);
    const path = kind === 'lmstudio' ? '/api/v0/models' : '/api/tags';
    let res: Response;
    try {
      res = await safeFetchLlm(`${root}${path}`, { signal: AbortSignal.timeout(10_000) });
    } catch {
      throw new HttpException({ error: `Could not reach local LLM server at ${root}` }, 502);
    }
    if (!res.ok) throw new HttpException({ error: `Local LLM server error (${res.status})` }, 502);
    const data = (await res.json()) as LocalModelsResponse;
    return { models: kind === 'lmstudio' ? lmStudioModels(data) : ollamaModels(data) };
  }

  /**
   * Start a streamed pull. Returns the upstream NDJSON body so the controller can
   * pipe Ollama's progress lines straight to the client.
   *
   * Ollama only: LM Studio downloads models through its own app (or the `lms` CLI)
   * and exposes no download endpoint on its REST API. Saying so is the honest answer
   * — the alternative would be a button that reports a 404 from somebody else's server.
   */
  async pull(baseUrl: string | undefined, model: string, provider?: string): Promise<ReadableStream<Uint8Array>> {
    if (this.resolveProvider(provider) === 'lmstudio') {
      throw new HttpException(
        { error: 'LM Studio has no download API — add the model in the LM Studio app (or run `lms get <model>`), then refresh.' },
        400,
      );
    }
    if (!model?.trim()) throw new HttpException({ error: 'model is required' }, 400);
    const root = this.serverRoot(baseUrl, 'local');
    let res: Response;
    try {
      res = await safeFetchLlm(`${root}/api/pull`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: model.trim(), stream: true }),
      });
    } catch {
      throw new HttpException({ error: `Could not reach local LLM server at ${root}` }, 502);
    }
    if (!res.ok || !res.body) throw new HttpException({ error: `Pull failed (${res.status})` }, 502);
    return res.body;
  }
}

/** Both servers' list shapes — Ollama answers `{models}`, LM Studio `{data}`. */
interface LocalModelsResponse {
  models?: { name?: string; size?: number }[];
  data?: { id?: string; type?: string }[];
}

function ollamaModels(data: LocalModelsResponse): LocalModel[] {
  return (data.models ?? []).map(m => ({ name: m.name ?? '', size: m.size ?? 0 })).filter(m => m.name);
}

/**
 * LM Studio lists every downloaded model, embedding models included, and reports no
 * size. An embedding model cannot answer a chat completion, so offering one would only
 * produce a confusing failure at import time — the `type` field says which is which
 * ('llm', 'vlm', 'embeddings'); an unfamiliar type is kept rather than hidden.
 */
function lmStudioModels(data: LocalModelsResponse): LocalModel[] {
  return (data.data ?? [])
    .filter(m => m.type !== 'embeddings')
    .map(m => ({ name: m.id ?? '', size: 0 }))
    .filter(m => m.name);
}
