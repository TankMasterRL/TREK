/** A single binary file (e.g. a PDF) sent natively to a multimodal provider. */
export interface LlmExtractionFile {
  mimeType: string;
  data: Buffer;
}

/** Everything a provider client needs to extract reservations from one document. */
export interface LlmExtractionInput {
  /** System instructions enumerating the schema.org shape (see llm-prompt.ts). */
  prompt: string;
  /** JSON Schema describing `{ reservations: KiReservation[] }`. */
  jsonSchema: object;
  model: string;
  baseUrl?: string;
  apiKey?: string;
  /** Pre-extracted text (text-like files, or text-only-model mode). */
  text?: string;
  /** Native binary (PDF) for multimodal providers. */
  file?: LlmExtractionFile;
}

/**
 * A provider client turns one document into raw schema.org reservation objects.
 * It returns the parsed `reservations` array (best-effort: `[]` on a malformed or
 * empty response, never throwing for content reasons). The caller validates and
 * maps via the shared kitinerary mapper.
 *
 * What it does throw for is the document never getting a fair reading: an
 * unreachable endpoint, a rejected request, a reply the configured token cap cut
 * short before a single reservation came back. Those have a remedy the operator
 * can act on, and llm-parse.service.ts turns the message into the warning shown
 * against the file — an empty array would say "nothing in this document" and
 * hide the reason.
 */
export interface LlmExtractionClient {
  extract(input: LlmExtractionInput): Promise<Record<string, unknown>[]>;
}
