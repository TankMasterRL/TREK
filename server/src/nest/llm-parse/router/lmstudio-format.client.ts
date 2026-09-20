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
 * LM Studio also exposes a native `/api/v0/chat/completions` with the same request shape
 * plus load/runtime stats in the response. Extraction needs none of those stats, and the
 * `/v1` path is the one LM Studio documents as stable, so that is the one used here.
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
