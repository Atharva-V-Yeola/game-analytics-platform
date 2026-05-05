import React, { useState } from 'react';

export default function DataViewer({ files, onRefresh }) {
  const [spinning, setSpinning] = useState(false);

  const handleRefresh = async () => {
    setSpinning(true);
    await onRefresh();
    setTimeout(() => setSpinning(false), 600);
  };

  return (
    <section className="data-section">
      <div className="data-section-header">
        <div className="data-section-title">
          <span className="data-icon">📁</span>
          Collected Data
          {files.length > 0 && (
            <span className="file-count-badge">{files.length} file{files.length > 1 ? 's' : ''}</span>
          )}
        </div>
        <button
          id="btn-refresh-data"
          className={`refresh-btn ${spinning ? 'spinning' : ''}`}
          onClick={handleRefresh}
          title="Refresh file list"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="23 4 23 10 17 10"/>
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
          </svg>
          Refresh
        </button>
      </div>

      {files.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">📭</div>
          <p>
            No CSV files yet.<br />
            Start a game and stop it — data will appear here.
          </p>
        </div>
      ) : (
        <ul className="file-list">
          {files.map((name, i) => (
            <li
              key={name}
              className="file-item"
              style={{ animationDelay: `${i * 40}ms` }}
            >
              <span className="file-icon">📄</span>
              <span className="file-name">{name}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
