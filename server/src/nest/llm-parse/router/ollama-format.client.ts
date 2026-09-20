/**
 * Minimal Ollama native-API client used by the extraction router.
 *
 * Why not the OpenAI-compatible `/v1/chat/completions` path the rest of llm-parse uses?
 * Ollama's `/v1` endpoint does NOT faithfully honour OpenAI's `response_format:{json_schema,strict}`
 * (it's passed through loosely — the schema and `strict` flag are effectively ignored).
 * Ollama's OWN `/api/chat` endpoint with a top-level `format: <jsonSchema>` is the path that
 * actually compiles the schema to a GBNF grammar and constrains token sampling. That hard
 * guarantee — valid, type-correct, all-required-fields JSON — is the router's foundation,
 * so the router talks to `/api/chat` directly. (LM Studio reaches the same guarantee through
 * its own OpenAI-compatible endpoint — see ./lmstudio-format.client.ts. Cloud providers
 * enforce via their own strict tool/response_format and keep using the existing clients.)
 */

import { postEnforced, toNativeBase, type EnforcedExtractInput } from './enforced';

/**
 * Run one schema-constrained chat completion against Ollama's native `/api/chat`.
 * Returns the parsed JSON object (constrained to `schema`), or null if the request
 * produced unparseable output.
 */
export async function extractEnforced(input: EnforcedExtractInput): Promise<Record<string, unknown> | null> {
  const url = `${toNativeBase(input.baseUrl)}/api/chat`;
  const body = {
    model: input.model,
    stream: false,
    format: input.schema,
    // Disable "thinking" for hybrid/reasoning models (Qwen3, etc.): the reasoning tokens
    // collide with the format-grammar constraint here — they produce unparseable output and
    // blow the latency budget on CPU. Ollama ignores this for non-thinking models, so it's safe.
    think: false,
    // Keep the model resident a while so back-to-back imports don't pay the cold load.
    keep_alive: '30m',
    options: { temperature: 0, num_predict: input.numPredict ?? 512, num_ctx: input.numCtx ?? 8192 },
    messages: [
      { role: 'system', content: input.system },
      { role: 'user', content: input.user },
    ],
  };

  return postEnforced(
    url,
    body,
    input.apiKey,
    'Ollama /api/chat',
    data => (data as { message?: { content?: string } })?.message?.content,
  );
}
