import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { LaunchSettings } from '../components.jsx'

export default function Favorites({ onLaunched, onEdit, notify }) {
  const [favorites, setFavorites] = useState(null)
  const [error, setError] = useState('')
  const [launching, setLaunching] = useState(null)
  const [archivePath, setArchivePath] = useState('')
  const [confirmArchive, setConfirmArchive] = useState(null)
  const [archiving, setArchiving] = useState(null)
  const [archived, setArchived] = useState('')

  useEffect(() => {
    let alive = true
    api.favorites().then(r => { if (alive) { setFavorites(r.favorites); setArchivePath(r.archive_path) } })
      .catch(err => { if (alive) setError(err.message) })
    return () => { alive = false }
  }, [])

  const launch = async (favorite) => {
    setLaunching(favorite.id)
    try {
      const server = await api.launchFavorite(favorite.id)
      notify(`Launching “${favorite.name}”.`)
      if (server.history_save_error) notify(server.history_save_error, true)
      await onLaunched(server)
    } catch (err) {
      notify(err.message, true)
    } finally {
      setLaunching(null)
    }
  }

  const archive = async (favorite) => {
    setArchiving(favorite.id)
    try {
      const result = await api.archiveFavorite(favorite.id)
      setFavorites(current => current.filter(f => f.id !== favorite.id))
      setArchived(`“${favorite.name}” was removed from Favorites. A recovery copy is saved in ${result.archive_path}.`)
      setConfirmArchive(null)
    } catch (err) {
      notify(err.message, true)
    } finally {
      setArchiving(null)
    }
  }

  if (error) return <p role="alert">{error}</p>
  if (favorites === null) return <p className="muted">Loading favorites…</p>
  return (
    <div className="stack">
      {archived && <p className="archive-notice" role="status">{archived}</p>}
      {!favorites.length && <div className="empty">No favorites yet. Open Servers and choose Save Launch Settings to Favorites.</div>}
      {favorites.map(f => (
        <section className="section" key={f.id}>
          <div className="section-head">
            <div className="favorite-heading">
              <h2>{f.name}</h2>
              <p className="small muted">{f.repo_id} · {f.engine_mode}</p>
            </div>
            <div className="row favorite-actions">
            <button className="btn btn-ghost sm" disabled={launching !== null || archiving !== null} onClick={() => onEdit(f)}>Edit</button>
            <button className="btn btn-ghost sm" disabled={launching !== null || archiving !== null} onClick={() => setConfirmArchive(f.id)}>Archive</button>
            <button className="btn btn-primary sm" disabled={launching !== null || archiving !== null} onClick={() => launch(f)}>
              {launching === f.id ? 'Launching…' : 'Launch'}
            </button>
            </div>
          </div>
          {confirmArchive === f.id && <div className="archive-notice" role="alert">
            <p>Remove this favorite? Its settings will be saved in <code>{archivePath}</code> on the launcher computer. You can recover them from that file later.</p>
            <div className="row" style={{ marginTop: 10 }}>
              <button className="btn btn-primary sm" disabled={archiving !== null} onClick={() => archive(f)}>{archiving === f.id ? 'Archiving…' : 'Archive favorite'}</button>
              <button className="btn btn-ghost sm" disabled={archiving !== null} onClick={() => setConfirmArchive(null)}>Cancel</button>
            </div>
          </div>}
          <div className="saved-settings"><LaunchSettings config={f.config} /></div>
        </section>
      ))}
    </div>
  )
}
