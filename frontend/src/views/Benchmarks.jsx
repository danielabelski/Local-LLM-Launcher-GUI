import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { LaunchSettings } from '../components.jsx'

const number = value => value == null ? '—' : value.toLocaleString(undefined, { maximumFractionDigits: 2 })

export default function Benchmarks() {
  const [benchmarks, setBenchmarks] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let alive = true
    api.benchmarks().then(r => { if (alive) setBenchmarks(r.benchmarks) })
      .catch(err => { if (alive) setError(err.message) })
    return () => { alive = false }
  }, [])

  if (error) return <p role="alert">{error}</p>
  if (benchmarks === null) return <p className="muted">Loading benchmarks…</p>
  if (!benchmarks.length) return <div className="empty">No benchmarks yet. Run a test chat on Servers, then choose Benchmark to save measurements from the log.</div>
  return (
    <div className="stack">
      <p className="small muted">Snapshots of the full server log at capture time. Prefill is prompt processing. llama.cpp rates use tokens divided by measured time; vLLM rates average logged intervals, including reported zeros. Workloads and measurement methods can differ. — means unavailable.</p>
      <div className="section benchmark-scroll" tabIndex={0} role="region" aria-label="Benchmark results">
        <table className="data-table benchmark-table">
          <caption className="sr-only">Recorded model performance and launch settings</caption>
          <thead><tr>
            <th scope="col">Model / settings</th><th scope="col">Prefill TPS</th><th scope="col">Generation TPS</th>
            <th scope="col">Prompt tokens</th><th scope="col">Generated tokens</th>
            <th scope="col">Prefill / generation time (ms)</th><th scope="col">Samples (prefill / generation)</th><th scope="col">Measurement / captured</th>
          </tr></thead>
          <tbody>{[...benchmarks].reverse().map(b => (
            <tr key={b.id}>
              <td className="benchmark-model"><strong>{b.repo_id}</strong><p className="small muted">{b.engine_mode}</p>
                <details><summary>Launch settings</summary><LaunchSettings config={b.config} />
                  {b.config_note && <p className="small muted">Some settings are omitted because they contain credentials or invalid quoting.</p>}
                </details>
              </td>
              <td className="mono">{number(b.metrics.prefill_tps)}</td>
              <td className="mono">{number(b.metrics.generation_tps)}</td>
              <td>{number(b.metrics.prompt_tokens)}</td><td>{number(b.metrics.generated_tokens)}</td>
              <td>{number(b.metrics.prefill_ms)} / {number(b.metrics.generation_ms)}</td>
              <td>{b.metrics.prefill_samples} / {b.metrics.generation_samples}</td>
              <td>{b.metrics.basis}<p className="small muted">{new Date(b.created_at).toLocaleString()}</p></td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  )
}
