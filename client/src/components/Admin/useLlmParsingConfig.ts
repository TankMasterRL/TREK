import { useCallback, useEffect, useMemo, useState } from 'react'
import { adminApi } from '../../api/client'
import { useToast } from '../shared/Toast'
import { LLM_PROVIDER_META, recommendedModelsFor } from './llmProviders'

const FALLBACK_META = LLM_PROVIDER_META[0]

export interface LlmParsingConfigAddon {
  id: string
  config?: Record<string, unknown>
}

/**
 * Instance-wide AI-parsing config — the ONE logic path behind both admin shells
 * (the desktop tile shelf and the phone sheet render their own markup over this
 * state). When set, it applies to the whole instance and overrides every user's
 * personal config (see the server's llm-config.resolver.ts).
 *
 * The API key is masked on read; the mask is sent back unchanged so the server keeps
 * the stored key. For a self-hosted provider the hook also lists the models installed on
 * that server and downloads a recommended one — Ollama through `/api/tags` + `/api/pull`,
 * LM Studio through its native v1 REST API — both behind the admin route, which takes the
 * provider so it knows which server to ask and which id spelling that one downloads by.
 */
export function useLlmParsingConfig(addon: LlmParsingConfigAddon) {
  const toast = useToast()
  const cfg = useMemo(() => (addon.config ?? {}) as Record<string, unknown>, [addon.config])
  const [provider, setProvider] = useState<string>((cfg.provider as string) ?? 'local')
  const [model, setModel] = useState<string>((cfg.model as string) ?? '')
  const [baseUrl, setBaseUrl] = useState<string>((cfg.baseUrl as string) ?? '')
  const [apiKey, setApiKey] = useState<string>((cfg.apiKey as string) ?? '')
  const [saving, setSaving] = useState(false)

  // Self-hosted model management.
  const [installed, setInstalled] = useState<string[]>([])
  const [modelsErr, setModelsErr] = useState('')
  const [loadingModels, setLoadingModels] = useState(false)
  const [pulling, setPulling] = useState<string | null>(null)
  const [pullPct, setPullPct] = useState(0)
  const [pullStatus, setPullStatus] = useState('')

  const meta = LLM_PROVIDER_META.find(p => p.value === provider) ?? FALLBACK_META
  const effectiveUrl = baseUrl.trim() || meta.defaultBaseUrl || ''
  // Resolved here rather than in each shell: the two servers spell the same model
  // differently, and that lookup is logic, so both shells render one ready-made list.
  const recommended = useMemo(() => recommendedModelsFor(meta.value), [meta.value])
  const isInstalled = (id: string) => installed.some(n => n === id || n.startsWith(id + ':') || n.startsWith(id))

  const loadModels = useCallback(async () => {
    if (!meta.selfHosted) return
    setLoadingModels(true)
    setModelsErr('')
    try {
      const res = await adminApi.llmLocalModels(effectiveUrl, meta.value)
      setInstalled(res.models.map(m => m.name))
    } catch (e: unknown) {
      setModelsErr(e instanceof Error ? e.message : 'Could not reach the local LLM server')
      setInstalled([])
    } finally {
      setLoadingModels(false)
    }
  }, [effectiveUrl, meta.selfHosted, meta.value])

  // Load the installed models when a self-hosted provider is active. Deliberately keyed
  // on the provider alone: typing in the Base URL field would otherwise fire a request
  // per keystroke, so the field refetches on blur instead (see `loadModels` above).
  useEffect(() => {
    if (meta.selfHosted) loadModels()
    else { setInstalled([]); setModelsErr('') }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider])

  const pull = async (id: string) => {
    if (pulling) return
    setPulling(id)
    setPullPct(0)
    setPullStatus('starting…')
    try {
      await adminApi.llmLocalPull(effectiveUrl, id, (p) => {
        if (p.error) throw new Error(p.error)
        if (p.status) setPullStatus(p.status)
        if (p.total && p.completed != null) setPullPct(Math.round((p.completed / p.total) * 100))
      }, meta.value)
      toast.success('Model pulled')
      setModel(id)
      await loadModels()
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : 'Pull failed')
    } finally {
      setPulling(null)
      setPullPct(0)
      setPullStatus('')
    }
  }

  const save = async () => {
    setSaving(true)
    try {
      // Send the masked sentinel unchanged so the server keeps the stored key. A
      // provider with no endpoint of its own (Anthropic) stores no base URL, so
      // switching to it also clears one left over from another provider.
      await adminApi.updateAddon(addon.id, {
        config: {
          provider,
          model: model.trim(),
          baseUrl: meta.defaultBaseUrl ? baseUrl.trim() : '',
          apiKey,
          multimodal: cfg.multimodal === true,
        },
      })
      toast.success('Saved')
    } catch {
      toast.error('Failed to save')
    } finally {
      setSaving(false)
    }
  }

  return {
    provider, setProvider,
    model, setModel,
    baseUrl, setBaseUrl,
    apiKey, setApiKey,
    meta,
    saving, save,
    installed, isInstalled, modelsErr, loadingModels, loadModels,
    recommended, pulling, pullPct, pullStatus, pull,
  }
}
