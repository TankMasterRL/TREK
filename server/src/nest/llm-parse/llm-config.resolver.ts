import { ADDON_IDS } from '../../addons';
import { AddonsService } from '../addons/addons.service';
import {
  decryptLlmApiKey,
  isSelfHostedLlmProvider,
  LLM_PROVIDERS,
  type LlmProvider,
  type ResolvedLlmConfig,
} from './llm-config';
import { DatabaseService } from '../database/database.service';
import { SettingsService } from '../settings/settings.service';
import { Injectable } from '@nestjs/common';

function asProvider(v: unknown): LlmProvider | null {
  return typeof v === 'string' && (LLM_PROVIDERS as string[]).includes(v) ? (v as LlmProvider) : null;
}

/**
 * Resolves the effective LLM config for a user, gated by the addon. Injectable
 * (settings come from SettingsService); the addon-row read and the addon gate
 * still go through the legacy db/adminService seams until their waves migrate.
 */
@Injectable()
export class LlmConfigResolver {
  constructor(
    private readonly settings: SettingsService,
    private readonly dbService: DatabaseService,
    private readonly addons: AddonsService,
  ) {}

  /**
   * Resolve the effective LLM config for a user, gated by the addon.
   * Order: addon disabled → null; admin instance config wins; else per-user config;
   * else null. This is the single place the API key is decrypted, and the single
   * place that decides which endpoint the server is allowed to call (#1772).
   */
  resolve(userId: number): ResolvedLlmConfig | null {
    if (!this.addons.isAddonEnabled(ADDON_IDS.LLM_PARSING)) return null;
    return this.readInstanceConfig() ?? this.readUserConfig(userId);
  }

  /**
   * The admin's stored API key for the instance config, decrypted.
   *
   * Separate from resolve() because the admin management routes need it BEFORE
   * there is a resolvable config: the panel lists a self-hosted server's models
   * so the admin can pick one, so the model field is still empty at that point
   * and readInstanceConfig() answers null. A v1 LM Studio can be configured to
   * require a token (`Authorization: Bearer`), and it refuses the model list
   * without one — so the key is read on its own rather than inferred from a
   * config that is not finished yet.
   */
  instanceApiKey(): string | undefined {
    return decryptLlmApiKey(this.readInstanceConfigBlob()?.apiKey);
  }

  /** The `llm_parsing` addon's stored config JSON, or null when unset/unparseable. */
  private readInstanceConfigBlob(): Record<string, unknown> | null {
    const row = this.dbService.get<{ config?: string } | undefined>(
      'SELECT config FROM addons WHERE id = ?',
      ADDON_IDS.LLM_PARSING,
    );
    if (!row?.config) return null;
    try {
      return JSON.parse(row.config || '{}');
    } catch {
      return null;
    }
  }

  private readInstanceConfig(): ResolvedLlmConfig | null {
    const cfg = this.readInstanceConfigBlob();
    if (!cfg) return null;
    const provider = asProvider(cfg.provider);
    const model = typeof cfg.model === 'string' ? cfg.model.trim() : '';
    if (!provider || !model) return null;
    return {
      provider,
      model,
      baseUrl: typeof cfg.baseUrl === 'string' && cfg.baseUrl.trim() ? cfg.baseUrl.trim() : undefined,
      apiKey: decryptLlmApiKey(cfg.apiKey),
      multimodal: cfg.multimodal === true,
    };
  }

  private readUserConfig(userId: number): ResolvedLlmConfig | null {
    const settings = this.settings.getUserSettings(userId);
    const provider = asProvider(settings.llm_provider);
    const model = typeof settings.llm_model === 'string' ? settings.llm_model.trim() : '';
    if (!provider || !model) return null;

    // #1772: the address this server calls is instance configuration, never a
    // personal preference. The request leaves OUR network and safeFetchLlm
    // deliberately allows loopback/LAN so a self-hosted Ollama or LM Studio keeps working,
    // which is a reasonable trade for whoever runs the instance and a network
    // probe for anyone else. An instance has exactly one such address, so it
    // comes from the admin-set instance-wide defaults for EVERY user, including
    // an admin's own row. This is the choke point every consumer passes
    // (booking import and the plugin RPC surface), and the only place that also
    // catches values already sitting in the db.
    const endpoints = this.settings.getAdminUserDefaults();
    // A self-hosted provider ('local' = Ollama, 'lmstudio' = LM Studio) is an
    // endpoint choice too ("some address I name"), so without an admin-set
    // endpoint there is no config at all, never a silent redirect to a
    // different provider. The admin default must name the SAME provider: the
    // one address an instance has belongs to one kind of server, and Ollama's
    // native API is not LM Studio's.
    if (isSelfHostedLlmProvider(provider) && asProvider(endpoints.llm_provider) !== provider) return null;
    const baseUrl =
      typeof endpoints.llm_base_url === 'string' && endpoints.llm_base_url.trim()
        ? endpoints.llm_base_url.trim()
        : undefined;

    const apiKey = this.settings.getDecryptedUserSetting(userId, 'llm_api_key') ?? undefined;
    return {
      provider,
      model,
      baseUrl,
      apiKey,
      multimodal: settings.llm_multimodal === true,
    };
  }
}
