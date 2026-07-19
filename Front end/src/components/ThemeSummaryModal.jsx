import { useEffect } from "react";

export default function ThemeSummaryModal({ metric, storeName, onClose }) {
  useEffect(() => {
    const onKey = (event) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!metric || metric.empty) return null;

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="modal-panel theme-summary-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="theme-summary-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <p className="modal-kicker">{storeName}</p>
            <h2 id="theme-summary-title">{metric.label}</h2>
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="theme-summary-score">
          <span className="theme-summary-score-value">{metric.display}</span>
          <span className="theme-summary-score-meta">
            From {metric.mention_count} matching review
            {metric.mention_count === 1 ? "" : "s"}
          </span>
        </div>

        <section className="modal-section">
          <h3>Theme summary</h3>
          <p>
            {metric.summary?.trim()
              ? metric.summary
              : "No LLM summary is available for this theme yet."}
          </p>
        </section>
      </div>
    </div>
  );
}
