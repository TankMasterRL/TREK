/**
 * What every schema-enforced extraction call has in common, whichever local
 * server answers it.
 *
 * The router's foundation is a HARD guarantee that the model's answer is valid,
 * type-correct JSON with every required field — not a request that it please
 * produce some. Each supported server reaches that guarantee through its own
 * endpoint (Ollama compiles a schema to a GBNF grammar on `/api/chat`, LM Studio
 * enforces `response_format: json_schema` on `/v1/chat/completions`), so the
 * request bodies differ and the plumbing around them does not: the same SSRF
 * guard, the same timeout, the same non-ok handling, the same lenient parse of a
 * model that wrapped its JSON in a fence anyway.
 *
 * That plumbing lives here once. A client module is then just "the body this
 * server wants, and where its answer sits".
 */

import { parseLenientJson } from '../lenient-json';
import { safeFetchLlm } from '../../../utils/ssrfGuard';
import { readEnv } from '../../../app-config';

export interface EnforcedExtractInput {
  /** Base URL as configured — the `/v1` suffix is added or stripped per server. */
  baseUrl: string;
  model: string;
  system: string;
  user: string;
  /** JSON Schema the output is constrained to (grammar-level). */
  schema: Record<string, unknown>;
  apiKey?: string;
  numPredict?: number;
  /** Context window. 8192 fits a typical multi-section booking; raise for long itineraries. */
  numCtx?: number;
}

/** One document → one schema-constrained object (or null when it was unreadable). */
export type EnforcedExtractor = (input: EnforcedExtractInput) => Promise<Record<string, unknown> | null>;

/** Strip trailing slashes and a `/v1` suffix — the root a native API hangs off. */
export function toNativeBase(baseUrl: string): string {
  // Trailing slashes come off as a scan, not /\/+$/: that pattern is unanchored at the
  // front, so the engine restarts it at every slash of a run and rescans to the end each
  // time (quadratic — 25s on 200k). Same result: the maximal trailing run goes.
  let end = baseUrl.length;
  while (end > 0 && baseUrl[end - 1] === '/') end--;
  return baseUrl.slice(0, end).replace(/\/v1$/, '');
}

/** The OpenAI-compatible base (`…/v1`), whether or not the config already spelled it. */
export function toOpenAiBase(baseUrl: string): string {
  return `${toNativeBase(baseUrl)}/v1`;
}

/** Tag names a reasoning model wraps its chain of thought in. */
const THINK_TAGS = ['think', 'thinking', 'reasoning'];

/**
 * Drop a reasoning block printed ahead of the answer, if there is one.
 *
 * Deliberately an index scan rather than a regex: the obvious pattern for this
 * (`^\s*<(think|…)>[\s\S]*?<\/\1>`) puts a lazy any-character run between two
 * literals on untrusted model output, and this runs on every extraction.
 * `indexOf` cannot backtrack at all, and reads no worse.
 */
function stripThinkBlock(content: string): string {
  const head = content.trimStart();
  const lower = head.toLowerCase();
  for (const tag of THINK_TAGS) {
    const open = `<${tag}>`;
    if (!lower.startsWith(open)) continue;
    const close = `</${tag}>`;
    const end = lower.indexOf(close, open.length);
    // An unterminated block is not a block — leave the content for the parser,
    // which answers null for it either way.
    if (end !== -1) return head.slice(end + close.length);
  }
  return content;
}

/**
 * POST one enforced request and read the model's JSON back.
 *
 * `label` names the endpoint in the thrown error, so a failure says which server
 * refused rather than "the LLM". `readContent` pulls the answer out of that
 * server's response envelope. A non-ok status throws (the router turns it into a
 * warning on the parse result); unreadable content is `null`, because a model
 * that answered with prose is not an outage.
 */
export async function postEnforced(
  url: string,
  body: unknown,
  apiKey: string | undefined,
  label: string,
  readContent: (data: unknown) => string | undefined | null,
): Promise<Record<string, unknown> | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), readEnv().integrations.llmTimeoutMs);
  let res: Response;
  try {
    // baseUrl is user-configurable — guard it against the cloud-metadata range,
    // while still allowing a local/LAN model server.
    res = await safeFetchLlm(url, {
      method: 'POST',
      signal: controller.signal,
      headers: {
        'content-type': 'application/json',
        ...(apiKey ? { authorization: `Bearer ${apiKey}` } : {}),
      },
      body: JSON.stringify(body),
    });
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`${label} failed (${res.status}): ${detail.slice(0, 200)}`);
  }

  const content = readContent(await res.json());
  // A reasoning model that cannot be told to stop thinking prints the chain of
  // thought ahead of the constrained answer. Ollama is asked for `think: false`
  // and LM Studio applies the schema to the visible channel, so this is the rare
  // leftover rather than the rule — drop a leading block instead of handing
  // parseLenientJson a document it will only return null for.
  const payload = typeof content === 'string' ? stripThinkBlock(content) : content;
  const parsed = parseLenientJson(payload);
  return parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : null;
}
