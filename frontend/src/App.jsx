import React, { useState, useEffect, useCallback } from 'react';
import GameCard from './components/GameCard';
import DataViewer from './components/DataViewer';

const API = '/api/games';
const POLL_MS = 2500;

export default function App() {
  const [games, setGames]     = useState([]);
  const [files, setFiles]     = useState([]);
  const [loading, setLoading] = useState({}); // gameId → true/false
  const [error, setError]     = useState(null);

  /* ── Fetch helpers ───────────────────────────────────────────── */
  const fetchGames = useCallback(async () => {
    try {
      const res = await fetch(API);
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      setGames(await res.json());
    } catch (e) {
      setError('Cannot reach backend — is Spring Boot running on :8080?');
    }
  }, []);

  const fetchFiles = useCallback(async () => {
    try {
      const res = await fetch(`${API}/data`);
      if (res.ok) setFiles(await res.json());
    } catch { /* silent */ }
  }, []);

  /* ── Initial load + polling ──────────────────────────────────── */
  useEffect(() => {
    fetchGames();
    fetchFiles();
    const id = setInterval(() => { fetchGames(); fetchFiles(); }, POLL_MS);
    return () => clearInterval(id);
  }, [fetchGames, fetchFiles]);

  /* ── Auto-dismiss error ──────────────────────────────────────── */
  useEffect(() => {
    if (!error || error.includes("Network error")) return;
    const t = setTimeout(() => setError(null), 5000);
    return () => clearTimeout(t);
  }, [error]);

  /* ── Game control ────────────────────────────────────────────── */
  const controlGame = async (id, action) => {
    setLoading(prev => ({ ...prev, [id]: true }));
    try {
      const res = await fetch(`${API}/${id}/${action}`, { method: 'POST' });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setError(body.error || `Failed to ${action} game`);
      }
    } catch {
      setError('Network error — cannot reach backend');
    } finally {
      setLoading(prev => ({ ...prev, [id]: false }));
      fetchGames();
      if (action === 'stop') setTimeout(fetchFiles, 1500); // let CSV flush
    }
  };

  const runningCount = games.filter(g => g.running).length;

  return (
    <div className="app">
      {/* ── Header ───────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-brand">
          <div className="header-icon">🎮</div>
          <div>
            <h1>Game Analytics Platform</h1>
            <p className="header-subtitle">Local AI-powered sports analytics — offline</p>
          </div>
        </div>
        <div className="header-status">
          <div className="status-dot-header" />
          {runningCount > 0
            ? `${runningCount} game${runningCount > 1 ? 's' : ''} running`
            : 'All idle'}
        </div>
      </header>

      {/* ── Games ────────────────────────────────────────────────── */}
      <p className="section-title">Available Games</p>
      {games.length === 0 ? (
        <div className="empty-state" style={{ marginBottom: 48 }}>
          <div className="empty-icon">⏳</div>
          <p>Connecting to backend…<br />Make sure Spring Boot is running on port 8080.</p>
        </div>
      ) : (
        <div className="games-grid">
          {games.map(game => (
            <GameCard
              key={game.id}
              game={game}
              busy={!!loading[game.id]}
              onStart={() => controlGame(game.id, 'start')}
              onStop={() => controlGame(game.id, 'stop')}
            />
          ))}
        </div>
      )}

      {/* ── Data Viewer ──────────────────────────────────────────── */}
      <DataViewer files={files} onRefresh={fetchFiles} />

      {/* ── Error Toast ──────────────────────────────────────────── */}
      {error && (
        <div className="error-toast">
          ⚠️ {error}
        </div>
      )}
    </div>
  );
}
