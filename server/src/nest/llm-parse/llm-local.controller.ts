import { Controller, Get, Post, Query, Body, Res, UseGuards } from '@nestjs/common';
import type { Response } from 'express';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { AdminGuard } from '../auth/admin.guard';
import { LlmLocalService } from './llm-local.service';
import { LlmLocalPullDto } from './llm-local.dto';
import { ManagedForbidden } from '../common/managed';

/**
 * Admin-only management of a self-hosted LLM server: list installed models, and pull
 * new ones where the server has an API for it. Used by the AI-parsing addon config UI.
 *
 * `provider` selects which server is being managed ('local' = Ollama, 'lmstudio' =
 * LM Studio). It is optional and defaults to Ollama, so a client that predates LM
 * Studio support keeps working unchanged.
 */
@Controller('api/admin/llm/local')
@UseGuards(JwtAuthGuard, AdminGuard)
export class LlmLocalController {
  constructor(private readonly local: LlmLocalService) {}

  @ManagedForbidden('the model list belongs to the operator runtime, not to one instance')
  @Get('models')
  models(@Query('baseUrl') baseUrl?: string, @Query('provider') provider?: string) {
    return this.local.listModels(baseUrl, provider);
  }

  /**
   * Stream a model pull. Proxies Ollama's NDJSON progress lines
   * ({ status, total?, completed? }) straight to the client, which reads the
   * response body to render a progress bar. Uses @Res() to stream manually.
   *
   * LM Studio has no download API and the service answers 400 for it — thrown before
   * anything is written, so the client still gets the standard `{ error }` envelope.
   */
  @ManagedForbidden('pulling a model spends the operator disk and GPU from inside a customer instance')
  @Post('pull')
  async pull(@Body() body: LlmLocalPullDto, @Res() res: Response): Promise<void> {
    const stream = await this.local.pull(body?.baseUrl, body?.model ?? '', body?.provider);
    res.status(200);
    res.setHeader('Content-Type', 'application/x-ndjson');
    res.setHeader('Cache-Control', 'no-cache');
    const reader = stream.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } catch {
      // Upstream dropped mid-pull — close the response; the client surfaces it.
    } finally {
      res.end();
    }
  }
}
