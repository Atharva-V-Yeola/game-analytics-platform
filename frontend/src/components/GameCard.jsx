import React from 'react';

export default function GameCard({ game, busy, onStart, onStop }) {
  const isRunning = game.running;

  return (
    <div className={`game-card ${isRunning ? 'running' : ''}`}>
      <div className="card-header">
        <div>
          <div className="card-title">{game.name}</div>
        </div>
        <div className={`status-badge ${isRunning ? 'running' : 'idle'}`}>
          <span className="status-dot" />
          {isRunning ? 'Running' : 'Idle'}
        </div>
      </div>

      <p className="card-description">{game.description}</p>

      <div className="card-footer">
        {isRunning ? (
          <button
            id={`btn-stop-${game.id}`}
            className="btn btn-stop"
            onClick={onStop}
            disabled={busy}
          >
            {busy ? <span className="btn-spinner" /> : <span className="btn-icon">⏹</span>}
            Stop Game
          </button>
        ) : (
          <button
            id={`btn-start-${game.id}`}
            className="btn btn-start"
            onClick={onStart}
            disabled={busy}
          >
            {busy ? <span className="btn-spinner" /> : <span className="btn-icon">▶</span>}
            Start Game
          </button>
        )}

        {game.statusMessage && !isRunning && (
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {game.statusMessage}
          </span>
        )}
      </div>
    </div>
  );
}
