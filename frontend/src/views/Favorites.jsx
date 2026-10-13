import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { LaunchSettings } from '../components.jsx'

export default function Favorites({ onLaunched, notify }) {
  const [favorites, setFavorites] = useState(null)
  const [error, setError] = useState('')
  const [launching, setLaunching] = useState(null)

  useEffect(() => {
    let alive = true
    api.favorites().then(r => { if (alive) setFavorites(r.favorites) })
      .catch(err => { if (alive) setError(err.message) })
    return () => { alive = false }
  }, [])

  const launch = async (favorite) => {
    setLaunching(favorite.id)
    try {
      const server = await api.launchFavorite(favorite.id)
      notify(`Launching “${favorite.name}”.`)
      await onLaunched(server)
    } catch (err) {
      notify(err.message, true)
    } finally {
      setLaunching(null)
    }
  }

  if (error) return <p role="alert">{error}</p>
  if (favorites === null) return <p className="muted">Loading favorites…</p>
  if (!favorites.length) return <div className="empty">No favorites yet. Open Servers and choose Save Launch Settings to Favorites.</div>
  return (
    <div className="stack">
      {favorites.map(f => (
        <section className="section" key={f.id}>
          <div className="section-head">
            <div className="favorite-heading">
              <h2>{f.name}</h2>
              <p className="small muted">{f.repo_id} · {f.engine_mode}</p>
            </div>
            <button className="btn btn-primary sm" disabled={launching !== null} onClick={() => launch(f)}>
              {launching === f.id ? 'Launching…' : 'Launch'}
            </button>
          </div>
          <div className="saved-settings"><LaunchSettings config={f.config} /></div>
        </section>
      ))}
    </div>
  )
}
