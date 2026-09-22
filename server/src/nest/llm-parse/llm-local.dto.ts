import { createZodDto } from 'nestjs-zod';
import { z } from 'zod';

/**
 * The model-pull body. Every field is optional because the service supplies its own
 * defaults for a missing baseUrl/provider and reports an empty model itself. `provider`
 * names which server is being managed ('local' = Ollama, 'lmstudio' = LM Studio); an
 * unknown or absent value means Ollama, which is what this route answered for before
 * LM Studio was supported.
 */
export class LlmLocalPullDto extends createZodDto(
  z.looseObject({
    baseUrl: z.string().optional(),
    model: z.string().optional(),
    provider: z.string().optional(),
  }),
) {}
