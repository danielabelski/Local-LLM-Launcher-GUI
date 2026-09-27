import { useEffect, useState } from 'react'
import { api } from '../api.js'

const LLAMA_INSTALL = {
  linux: `# Easiest: prebuilt binary via your package manager, or build with CUDA:
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
# binary lands at llama.cpp/build/bin/llama-server`,
  mac: `# Homebrew (Apple Silicon — Metal acceleration included):
brew install llama.cpp`,
}

export default function Settings({ hardware, notify }) {
  const [settings, setSettings] = useState(null)
  const [token, setToken] = useState('')
  const [clearToken, setClearToken] = useState(false)
  const [folders, setFolders] = useState('')
  const [llamaPath, setLlamaPath] = useState('')
  const [vllmPath, setVllmPath] = useState('')
  const [checks, setChecks] = useState({})
  const [checking, setChecking] = useState(null)
  const [job, setJob] = useState(null)
  const [updateError, setUpdateError] = useState('')
  const [lanAccess, setLanAccess] = useState(false)
  const [about, setAbout] = useState(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    api.settings().then((s) => {
      setSettings(s)
      setFolders((s.gguf_folders ?? []).join('\n'))
      setLlamaPath(s.llamacpp_path ?? '')
      setVllmPath(s.vllm_path ?? '')
      setLanAccess(!!s.lan_access)
    }).catch(() => setSettings({}))
    api.about().then(setAbout).catch(() => setAbout(null))
  }, [])

  useEffect(() => {
    // Status is local only. Upstream checks and builds require a button click.
    let cancelled = false
    const poll = async () => {
      try {
        const status = await api.updateStatus()
        if (!cancelled) setJob(status)
      } catch { /* LAN viewers cannot use the local updater. Check explains why. */ }
    }
    poll()
    const timer = setInterval(poll, 2000)
    return () => { cancelled = true; clearInterval(timer) }
  }, [])

  useEffect(() => {
    if (job?.state !== 'complete') return
    if (job.engine === 'llamacpp') setLlamaPath(job.executable)
    else setVllmPath(job.executable)
  }, [job?.state, job?.executable, job?.engine])

  const checkUpdate = async (engine) => {
    setChecking(engine)
    setUpdateError('')
    try {
      const result = await api.checkUpdate(engine)
      setChecks((previous) => ({ ...previous, [engine]: result }))
    } catch (err) { setUpdateError(err.message) }
    finally { setChecking(null) }
  }

  const startUpdate = async (check) => {
    setChecking(check.engine)
    setUpdateError('')
    try { setJob(await api.startUpdate(check.check_id)) }
    catch (err) { setUpdateError(err.message) }
    finally { setChecking(null) }
  }

  const save = async (e) => {
    e.preventDefault()
    setSaving(true)
    try {
      const updated = await api.saveSettings({
        ...(clearToken ? { hf_token: null } : token ? { hf_token: token } : {}),
        gguf_folders: folders.split('\n').map((f) => f.trim()).filter(Boolean),
        llamacpp_path: llamaPath || null,
        vllm_path: vllmPath || null,
        lan_access: lanAccess,
      })
      setSettings(updated)
      setToken('')
      setClearToken(false)
      notify('Settings saved.')
    } catch (err) {
      notify(err.message, true)
    } finally {
      setSaving(false)
    }
  }

  const isMac = !!hardware?.apple_silicon
  const llamaFound = !!hardware?.engines?.llamacpp_path

  return (
    <>
      <form className="section" onSubmit={save}>
        <div className="section-head">
          <div>
            <div className="section-title">Settings</div>
            <div className="section-subtitle">Configure tokens, paths, and preferences</div>
          </div>
          <button className="btn btn-primary" disabled={saving || job?.state === 'running'}>{saving ? 'Saving…' : 'Save settings'}</button>
        </div>

        <div style={{ padding: '14px 20px' }} className="stack">
          <div>
            <h3>Hugging Face access token</h3>
            <p className="small muted" style={{ margin: '4px 0 8px' }}>
              Only needed for "gated" models (like Meta's Llama) where you must accept a license
              first. Create one at{' '}
              <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
                huggingface.co/settings/tokens
              </a>{' '}
              — a "read" token is enough. It is stored only on this computer.
            </p>
            <input type="password" aria-label="Hugging Face access token" value={token}
              onChange={(e) => { setToken(e.target.value); setClearToken(false) }}
              placeholder={settings?.hf_token_set ? 'Saved (hidden)' : 'hf_…'}
              style={{ width: '100%', maxWidth: 420 }} />
            {settings?.hf_token_set && (
              <label className="row small" style={{ marginTop: 8 }}>
                <input type="checkbox" checked={clearToken}
                  onChange={(e) => { setClearToken(e.target.checked); setToken('') }} />
                Remove saved token when saving
              </label>
            )}
          </div>

          <div>
            <h3>Extra GGUF folders</h3>
            <p className="small muted" style={{ margin: '4px 0 8px' }}>
              If you keep .gguf model files outside the standard download cache, list those
              folders here (one per line) and they'll show up on the Models tab.
            </p>
            <textarea rows={3} value={folders} onChange={(e) => setFolders(e.target.value)}
              placeholder={'/home/you/models\n/mnt/storage/gguf'}
              style={{ width: '100%', maxWidth: 560, fontFamily: 'var(--font-mono)', fontSize: 12.5 }} />
          </div>

          <div>
            <h3>llama.cpp location</h3>
            <p className="small muted" style={{ margin: '4px 0 8px' }}>
              {llamaFound
                ? <>Found at <code className="mono">{hardware.engines.llamacpp_path}</code>. Set a path here only to use a different copy.</>
                : 'llama-server was not found automatically. If you installed it somewhere unusual, give the full path here.'}
            </p>
            <input value={llamaPath} onChange={(e) => setLlamaPath(e.target.value)}
              placeholder="/path/to/llama-server"
              style={{ width: '100%', maxWidth: 560, fontFamily: 'var(--font-mono)', fontSize: 12.5 }} />
          </div>

          <div>
            <h3>Native vLLM location</h3>
            <p className="small muted">Optional path to the vllm executable. Managed source builds fill this in; Docker launches use their existing image.</p>
            <input aria-label="Native vLLM location" value={vllmPath} onChange={(e) => setVllmPath(e.target.value)}
              placeholder="/path/to/venv/bin/vllm" style={{ width: '100%', maxWidth: 560 }} />
          </div>

          <div>
            <h3>Allow local network access</h3>
            <p className="small muted" style={{ margin: '4px 0 8px' }}>
              When off (default), model servers only listen on this computer (127.0.0.1) —
              the safe choice. Turn this on to let other devices on your local network reach
              the models, e.g. from your phone.
            </p>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input type="checkbox" checked={lanAccess} onChange={(e) => setLanAccess(e.target.checked)} />
              Expose servers to my local network
            </label>
          </div>
        </div>
      </form>

      <div className="section">
        <div className="section-head"><div className="section-title">Build current engine source</div></div>
        <div className="stack" style={{ padding: '14px 20px' }}>
          <p className="small muted">Optional Linux source builds download official upstream code and dependencies, then compile a separate copy.
            The latest source may contain unreleased changes. Builds can take a long time and several gigabytes.
            Only a successful build changes future launches; running servers and older copies are retained.
            Use a browser on this computer to update.</p>
          {['llamacpp', 'vllm'].map((engine) => {
            const check = checks[engine]
            return <div key={engine}>
              <h3>{engine === 'llamacpp' ? 'llama.cpp' : 'vLLM (native NVIDIA CUDA)'}</h3>
              <button type="button" className="btn" disabled={!!checking || job?.state === 'running'}
                onClick={() => checkUpdate(engine)}>{checking === engine ? 'Working…' : 'Check requirements and source'}</button>
              {check && <div className="small" style={{ marginTop: 8 }}>
                <div>Current executable: <code>{check.current_path || 'Automatic discovery'}</code></div>
                {check.current_revision && <div>Current managed source revision: <code>{check.current_revision}</code></div>}
                {check.revision && <div>Target source revision (commit): <code>{check.revision}</code></div>}
                <div>{check.backend.toUpperCase()} build · Up to {check.jobs} compiler jobs</div>
                {check.reasons.map((reason) => <p key={reason}>{reason}</p>)}
                {check.supported && <button type="button" className="btn btn-primary" disabled={!!checking || job?.state === 'running'}
                  onClick={() => startUpdate(check)}>Build and use this revision</button>}
              </div>}
            </div>
          })}
          {updateError && <p role="alert">{updateError}</p>}
          {job && job.state !== 'idle' && <div aria-live="polite">
            <strong>{job.engine}: {job.state}</strong><p>{job.stage}</p>
            {job.error && <p role="alert">{job.error}</p>}
            {job.log && <pre className="logbox" style={{ maxHeight: 240, overflow: 'auto', whiteSpace: 'pre-wrap' }}>{job.log}</pre>}
          </div>}
        </div>
      </div>

      {!llamaFound && (
        <div className="section">
          <div className="section-head">
            <div className="section-title">Installing llama.cpp</div>
          </div>
          <div style={{ padding: '14px 20px' }}>
            <p className="small muted" style={{ marginBottom: 10 }}>
              llama.cpp is the engine for GGUF models — the most beginner-friendly format.
              Install it with the commands below, then come back here.
            </p>
            <div className="logbox" style={{ height: 'auto', maxHeight: 200 }}>
              {isMac ? LLAMA_INSTALL.mac : LLAMA_INSTALL.linux}
            </div>
          </div>
        </div>
      )}

      <div className="section">
        <div className="section-head">
          <div className="section-title">About</div>
        </div>
        <div style={{ padding: '14px 20px' }}>
          <p className="small muted">
            Local-LLM-Launcher-GUI {about?.version && `v${about.version}`} — a friendly way to run
            large language models on your own computer.
          </p>
          {about?.credits?.map((c) => (
            <p key={c.url} className="small" style={{ padding: '3px 0' }}>
              <a href={c.url} target="_blank" rel="noreferrer">{c.name}</a>
              {c.note && <span className="muted"> — {c.note}</span>}
            </p>
          ))}
        </div>
      </div>
    </>
  )
}
