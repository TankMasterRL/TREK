/**
 * LM Studio client used by the extraction router.
 *
 * LM Studio serves an OpenAI-compatible API at `{base}/v1`, and unlike Ollama's `/v1`
 * shim it genuinely ENFORCES `response_format: { type: 'json_schema', json_schema: … }`:
 * the schema is compiled and applied to token sampling, so the answer is valid,
 * type-correct JSON with every required field. That is the same hard guarantee the
 * router gets from Ollama's native `format`, which is why LM Studio joins the router
 * rather than the best-effort single-shot client in ../clients/.
 *
 * So this is a separate client from ../clients/openai-compatible.client.ts on purpose.
 * That one is written for servers whose structured-output support is unknown: it
 * negotiates down through `json_object`, `max_completion_tokens` and a dropped
 * `temperature` as 400s come back. Here the server is known, one request is enough, and
 * the router's contract (one call per document, a schema that is actually enforced) holds.
 *
 * ## Why not the native v1 REST API
 *
 * LM Studio 0.4.0 released a native REST API at `/api/v1/*`, and the model management
 * next door (llm-local.service.ts) is written against it. Inference is the one place
 * that stays on the OpenAI-compatible path, for a reason that is not stylistic:
 * **`POST /api/v1/chat` has no `response_format`**. Its documented body is
 * `model` / `input` / `system_prompt` / `integrations` / sampling knobs / `reasoning` /
 * `context_length` / `store`, and nothing in it constrains the shape of the answer. Its
 * own strengths — stateful threads, MCP integrations, load progress events — are things
 * one-shot extraction has no use for, and it would cost the guarantee the whole router
 * is built on: asking a model for JSON is not the same as the server refusing to emit
 * anything else. `/v1/chat/completions` is where LM Studio applies a schema, so that is
 * where extraction runs, and this comment is here so the next reader does not "finish"
 * the migration by moving it.
 */

import { postEnforced, toOpenAiBase, type EnforcedExtractInput } from './enforced';

/**
 * Run one schema-constrained chat completion against LM Studio's `/v1/chat/completions`.
 * Returns the parsed JSON object (constrained to `schema`), or null if the request
 * produced unparseable output.
 */
export async function extractEnforcedLmStudio(
  input: EnforcedExtractInput,
): Promise<Record<string, unknown> | null> {
  const url = `${toOpenAiBase(input.baseUrl)}/chat/completions`;
  const body = {
    model: input.model,
    stream: false,
    // Extraction is deterministic — the same document must not parse differently twice.
    temperature: 0,
    max_tokens: input.numPredict ?? 512,
    // `strict` is what turns the schema from a hint into a constraint. LM Studio's own
    // docs print it as the string "true"; the boolean is what the OpenAI shape specifies
    // and what LM Studio's parser reads, so send the boolean.
    response_format: {
      type: 'json_schema',
      json_schema: { name: 'extraction', strict: true, schema: input.schema },
    },
    messages: [
      { role: 'system', content: input.system },
      { role: 'user', content: input.user },
    ],
  };

  return postEnforced(
    url,
    body,
    input.apiKey,
    'LM Studio /v1/chat/completions',
    data => (data as { choices?: { message?: { content?: string } }[] })?.choices?.[0]?.message?.content,
  );
}
