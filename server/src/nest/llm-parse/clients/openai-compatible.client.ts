import type { LlmExtractionClient, LlmExtractionInput } from '../llm-provider.interface';
import { isNuExtractModel, buildNuExtractUserText, nuExtractToKiReservations } from './nuextract';
import { parseLenientJson, toReservationList } from '../lenient-json';
import { safeFetchLlm } from '../../../utils/ssrfGuard';
import { readEnv } from '../../../app-config';

/**
 * The built-in cap, used when LLM_MAX_TOKENS is unset, and the value a raised
 * cap falls back to when a model turns out not to allow it.
 */
const DEFAULT_MAX_TOKENS = 4096;

/** What one attempt differs in. Each field is switched on by a 400 that asked for it. */
interface RequestShape {
  tokenParam: 'max_tokens' | 'max_completion_tokens';
  maxTokens: number;
  jsonObject: boolean;
  omitTemperature: boolean;
}

/**
 * Does this 400 body say the model refuses an explicit `temperature`?
 * OpenAI: "Unsupported value: 'temperature' does not support 0 with this model.
 * Only the default (1) value is supported."; Azure: "Unsupported parameter:
 * 'temperature' is not supported with this model."
 *
 * The bare word is not enough. Dropping temperature costs the deterministic
 * sampling every small local model depends on, and an error body can mention the
 * word while complaining about something else. When the phrasing is unfamiliar
 * nothing is lost — the request still falls through to the json_object retry,
 * exactly as it does today.
 */
function rejectsTemperature(detail: string): boolean {
  return /temperature/i.test(detail)
    && /unsupported|not supported|does not support|only the default/i.test(detail);
}

/**
 * Does this 400 body say the cap we asked for is larger than the model allows?
 *
 * The chat-completions schema puts NO maximum on either spelling of the field —
 * it is a plain nullable integer, and the real ceiling is the model's own output
 * budget (the completions schema spells the same rule out: "the token count of
 * your prompt plus `max_tokens` cannot exceed the model's context length").
 * Nothing static can know that number, so it is only ever learned from a
 * response: OpenAI answers "max_tokens is too large: 100000. This model supports
 * at most 16384 completion tokens.", vLLM/llama.cpp "This model's maximum
 * context length is 8192 tokens. However, you requested 10096 tokens ... Please
 * reduce the length", Mistral "max_tokens must be at most 8192".
 *
 * Both halves are required, for the reason rejectsTemperature needs both: a 400
 * that merely names the parameter is usually complaining about something else
 * (#1760's "use max_completion_tokens instead" names it and means the opposite),
 * and a wrong guess would quietly undo a cap the operator raised on purpose.
 */
function rejectsTokenCap(detail: string): boolean {
  return /max_tokens|max_completion_tokens|context length/i.test(detail)
    && /too large|too long|too many|at most|less than or equal|greater than|exceed|reduce the length|maximum context/i.test(detail);
}

/**
 * OpenAI-compatible chat-completions client. Covers both the "openai" cloud
 * provider and the "local" provider (Ollama / vLLM / llama.cpp / LM Studio),
 * which all expose `POST {baseUrl}/chat/completions`. Native binaries (PDF) are
 * sent as an OpenAI `file` content part; text goes as a text part. Uses the
 * global fetch (no SDK) to match the codebase's HTTP style.
 *
 * A NuExtract model (detected by id) takes a different request shape: the JSON
 * template inlined in a single user message, no system prompt and no
 * `response_format` (see ./nuextract.ts) — that's how the fine-tune expects to
 * be driven; the generic instruct path applies to every other model.
 *
 * Structured output is requested as `json_schema` first; servers that only
 * support `json_object` (DeepSeek, Mistral, some vLLM/llama.cpp) reject that
 * with a 400, so the request is retried once in `json_object` mode. Three
 * further 400s are answered the same way: `max_tokens` becomes
 * `max_completion_tokens` (#1760), "temperature is not supported" drops the
 * parameter (#2262), and a cap the model will not accept drops back to
 * DEFAULT_MAX_TOKENS.
 *
 * Those retries are a loop over what the server actually said, not a fixed
 * chain. A reasoning model rejects `max_tokens` AND `temperature`, the API names
 * only one parameter per response, and it may name either first — a chain of
 * one-shot ifs survives only one of the two orders. Each remedy applies at most
 * once, so this adds at most four extra requests.
 *
 * The size of the reply is capped by LLM_MAX_TOKENS (default DEFAULT_MAX_TOKENS).
 * The API leaves that field optional and unbounded — the ceiling belongs to the
 * model, not to the schema — so a document whose reservations do not fit in the
 * default needs the operator to raise it, and a value the model refuses falls
 * back rather than failing the import. A reply the cap did cut short is reported
 * instead of being swallowed as "no reservations found"; see extract().
 */
export class OpenAiCompatibleClient implements LlmExtractionClient {
  async extract(input: LlmExtractionInput): Promise<Record<string, unknown>[]> {
    // The lookbehind matches only the first slash of the trailing run. Without it the
    // engine retries from every slash, which is quadratic on a slash-heavy value.
    const base = (input.baseUrl ?? 'https://api.openai.com/v1').replace(/(?<!\/)\/+$/, '');
    const url = `${base}/chat/completions`;
    const nuextract = isNuExtractModel(input.model);

    const userContent: unknown[] = nuextract
      ? [{ type: 'text', text: buildNuExtractUserText(input.text ?? '') }]
      : [{ type: 'text', text: input.text ? `${USER_TEXT}\n\n${input.text}` : USER_TEXT }];
    // Only genuine images go natively (as image_url) — OpenAI-compatible servers
    // (notably Ollama) reject `file`/PDF content parts. PDFs reach this client as
    // pre-extracted text (see llm-parse.service.ts), never as bytes.
    if (!nuextract && input.file && input.file.mimeType.startsWith('image/')) {
      const b64 = input.file.data.toString('base64');
      userContent.push({
        type: 'image_url',
        image_url: { url: `data:${input.file.mimeType};base64,${b64}` },
      });
    }

    // The token cap is `max_tokens` for the chat-completions API and for every
    // local server (Ollama/vLLM/llama.cpp), but newer OpenAI models reject it
    // with a 400 and demand `max_completion_tokens`. Start with the broadly
    // supported spelling and swap on that specific rejection (#1760).
    const buildBody = (shape: RequestShape) => {
      const baseBody = {
        model: input.model,
        [shape.tokenParam]: shape.maxTokens,
        // Extraction is a deterministic task — Ollama defaults to 0.7, which makes
        // small models (NuExtract) drop fields or return empty. Pin to 0, and only
        // leave it out once a server has explicitly rejected the parameter (#2262).
        // The first attempt always carries it, and that is the only one a local
        // server ever sees.
        ...(shape.omitTemperature ? {} : { temperature: 0 }),
        // NuExtract wants the template (in the user turn) to be the only instruction
        // — a system prompt or a json_schema grammar derails it.
        messages: nuextract
          ? [{ role: 'user', content: userContent }]
          : [
              { role: 'system', content: input.prompt },
              { role: 'user', content: userContent },
            ],
      };
      if (nuextract) return baseBody;
      return {
        ...baseBody,
        response_format: shape.jsonObject
          ? { type: 'json_object' as const }
          : { type: 'json_schema' as const, json_schema: { name: 'reservations', schema: input.jsonSchema, strict: false } },
      };
    };

    // One read per document, so a change to the variable takes effect on the next
    // import rather than the next restart (readEnv() re-derives from process.env).
    const shape: RequestShape = {
      tokenParam: 'max_tokens',
      maxTokens: readEnv().integrations.llmMaxTokens,
      jsonObject: false,
      omitTemperature: false,
    };
    const tried = { tokenParam: false, temperature: false, tokenCap: false, jsonObject: false };

    let res = await this.send(url, buildBody(shape), input.apiKey);
    let detail = res.ok ? '' : await res.text().catch(() => '');

    // A 400 is the server naming the parameter it dislikes. Apply every remedy it
    // names, and only when it names none fall back to json_object — the unchanged
    // behaviour for servers that reject `json_schema` with an unspecific 400. The
    // system prompt already dictates the exact output shape and mentions JSON,
    // which json_object mode requires.
    while (!res.ok && res.status === 400) {
      let named = false;
      if (!tried.tokenParam && detail.includes('max_completion_tokens')) {
        shape.tokenParam = 'max_completion_tokens';
        tried.tokenParam = true;
        named = true;
      }
      if (!tried.temperature && rejectsTemperature(detail)) {
        shape.omitTemperature = true;
        tried.temperature = true;
        named = true;
      }
      // A raised cap the model will not honour: come back at the built-in
      // default rather than failing the import over a configuration value.
      // Guarded on actually being above it, so a server that answers this way at
      // the default doesn't buy an identical retry — that case falls through to
      // the json_object attempt below, as any other unspecific 400 does.
      if (!tried.tokenCap && shape.maxTokens > DEFAULT_MAX_TOKENS && rejectsTokenCap(detail)) {
        shape.maxTokens = DEFAULT_MAX_TOKENS;
        tried.tokenCap = true;
        named = true;
      }
      if (!named) {
        // NuExtract sends no response_format at all, so it has nothing to fall
        // back to and the 400 is final.
        if (tried.jsonObject || nuextract) break;
        shape.jsonObject = true;
        tried.jsonObject = true;
      }
      res = await this.send(url, buildBody(shape), input.apiKey);
      detail = res.ok ? '' : await res.text().catch(() => '');
    }

    if (!res.ok) {
      throw new Error(`LLM request failed (${res.status}): ${detail.slice(0, 300)}`);
    }

    const data = (await res.json()) as {
      choices?: { finish_reason?: string; message?: { content?: string } }[];
    };
    const choice = data.choices?.[0];
    const content = choice?.message?.content;
    const items = nuextract ? parseNuExtract(content) : parseReservations(content);

    // `length` is the spec's "the maximum number of tokens specified in the
    // request was reached": the reply is a fragment, and its JSON breaks off
    // mid-object. Lenient parsing recovers nothing from that, so the only
    // symptom an operator sees is an import that finds no reservations and
    // explains nothing — which is exactly the case LLM_MAX_TOKENS exists to fix,
    // and therefore the case worth naming. Whatever did parse is still returned:
    // a partial list beats discarding the document, with the cap in the log.
    if (choice?.finish_reason === 'length') {
      if (items.length === 0) {
        throw new Error(
          `the model hit the ${shape.maxTokens}-token response limit before returning any reservation — ` +
            'raise LLM_MAX_TOKENS, or use a model with a larger output budget',
        );
      }
      console.warn(
        `[llm-parse] Response truncated at the ${shape.maxTokens}-token limit; kept ${items.length} reservation(s). ` +
          'Raise LLM_MAX_TOKENS to get the rest.',
      );
    }
    return items;
  }

  private async send(url: string, body: unknown, apiKey?: string): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), readEnv().integrations.llmTimeoutMs);
    try {
      // baseUrl is user-configurable — guard it against pointing at the cloud
      // metadata endpoint, while still allowing a local/LAN Ollama.
      return await safeFetchLlm(url, {
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
  }
}

/** Parse a NuExtract response and map its flat template output to KiReservation nodes. */
function parseNuExtract(content: string | undefined | null): Record<string, unknown>[] {
  return nuExtractToKiReservations(parseLenientJson(content));
}

const USER_TEXT = 'Extract every travel reservation from the following document as schema.org JSON-LD.';

/** Tolerant parse: strip code fences, JSON(5).parse, pull `reservations`. `[]` on failure. */
function parseReservations(content: string | undefined | null): Record<string, unknown>[] {
  return toReservationList(parseLenientJson(content));
}
