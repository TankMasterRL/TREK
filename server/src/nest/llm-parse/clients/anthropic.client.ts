import type { LlmExtractionClient, LlmExtractionInput } from '../llm-provider.interface';
import { safeFetchLlm } from '../../../utils/ssrfGuard';
import { readEnv } from '../../../app-config';
import { toReservationList } from '../lenient-json';
import { UnreadableLlmResponse } from './openai-compatible.client';

const MAX_TOKENS = 8192;
const ANTHROPIC_VERSION = '2023-06-01';
const TOOL_NAME = 'emit_reservations';

/**
 * Anthropic Messages API client. Structured output via forced tool-use: a single
 * `emit_reservations` tool whose `input_schema` is the reservations schema, with
 * `tool_choice` forcing it — the documented, reliable way to get structured JSON.
 * PDFs go as native base64 `document` blocks (Anthropic reads scanned PDFs).
 * Raw fetch (no SDK) to match the codebase's HTTP style.
 *
 * The same client drives the 'anthropic-compatible' provider: any server that
 * speaks the Messages API at an operator-chosen base URL (a LiteLLM/OpenRouter
 * gateway, or another vendor's Anthropic-shaped endpoint). Such servers split on
 * how they read the key — Anthropic's `x-api-key` or the `Authorization: Bearer`
 * most gateways expect — so in compatible mode the key goes in both.
 */
export class AnthropicClient implements LlmExtractionClient {
  constructor(private readonly options: { compatible?: boolean } = {}) {}

  async extract(input: LlmExtractionInput): Promise<Record<string, unknown>[]> {
    const url = `${messagesBase(input.baseUrl)}/v1/messages`;
    const label = this.options.compatible ? 'Anthropic-compatible endpoint' : 'Anthropic';

    const content: unknown[] = [];
    if (input.file) {
      content.push({
        type: 'document',
        source: { type: 'base64', media_type: input.file.mimeType, data: input.file.data.toString('base64') },
      });
    }
    content.push({
      type: 'text',
      text: input.text ? `${USER_TEXT}\n\n${input.text}` : USER_TEXT,
    });

    const body = {
      model: input.model,
      max_tokens: MAX_TOKENS,
      system: input.prompt,
      tools: [
        {
          name: TOOL_NAME,
          description: 'Return the travel reservations extracted from the document.',
          input_schema: input.jsonSchema,
        },
      ],
      tool_choice: { type: 'tool', name: TOOL_NAME },
      messages: [{ role: 'user', content }],
    };

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), readEnv().integrations.llmTimeoutMs);
    let res: Response;
    try {
      // baseUrl is user-configurable — guard it against pointing at the cloud
      // metadata endpoint, while still allowing a local/LAN gateway.
      res = await safeFetchLlm(url, {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'content-type': 'application/json',
          'x-api-key': input.apiKey ?? '',
          ...(this.options.compatible && input.apiKey ? { authorization: `Bearer ${input.apiKey}` } : {}),
          'anthropic-version': ANTHROPIC_VERSION,
        },
        body: JSON.stringify(body),
      });
    } finally {
      clearTimeout(timer);
    }

    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new Error(`${label} request failed (${res.status}): ${detail.slice(0, 300)}`);
    }

    const data = (await res.json()) as {
      stop_reason?: string;
      content?: { type: string; name?: string; input?: { reservations?: unknown } }[];
    };

    if (data.stop_reason === 'refusal') {
      throw new Error(`${label} declined to process this document`);
    }
    // A run that hits the cap stops mid-tool-call, so what arrives is a fragment
    // of the list or nothing at all — and a forced tool that was not called left
    // no list either. Neither is "this document holds no booking", but both came
    // back as [] and reached the person as an empty preview with nothing in the
    // log: the #2375 symptom, on the provider the dropdown offers first.
    if (data.stop_reason === 'max_tokens') {
      throw new UnreadableLlmResponse(
        `the answer was cut off at the ${MAX_TOKENS}-token limit — the document is too long to extract in one pass`,
      );
    }

    const toolUse = data.content?.find(b => b.type === 'tool_use' && b.name === TOOL_NAME);
    if (!toolUse) throw new UnreadableLlmResponse(`the model answered without calling ${TOOL_NAME}`);
    return toReservationList(toolUse.input?.reservations);
  }
}

/**
 * The API root the `/v1/messages` path hangs off. Anthropic's own SDKs take the
 * root without `/v1`, but gateway docs routinely print it with one, and pasting
 * either must work — so a trailing `/v1` is dropped rather than doubled.
 */
function messagesBase(baseUrl: string | undefined): string {
  // The lookbehind pins the run to its own start. Without it `\/+$` restarts at every
  // slash of a trailing run that turns out not to end the string, rescanning to the
  // end each time; the assertion only rules out start positions the leftmost match
  // could never have used, so the trimmed result is unchanged.
  const base = (baseUrl ?? 'https://api.anthropic.com').replace(/(?<!\/)\/+$/, '');
  return base.endsWith('/v1') ? base.slice(0, -'/v1'.length) : base;
}

const USER_TEXT = 'Extract every travel reservation from the following document as schema.org JSON-LD.';
