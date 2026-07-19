import { useEffect } from "react";
import { getKpiGuide } from "../utils/kpiGuide";

function vsLabel(kpi) {
  if (kpi.vs_network === "on par") return "Same as network average";
  if (kpi.vs_network === "above") return "Above network average";
  if (kpi.vs_network === "below") return "Below network average";
  return "Compared with network average";
}

export default function KpiDetailModal({ kpi, storeName, onClose }) {
  const guide = getKpiGuide(kpi?.id);

  useEffect(() => {
    const onKey = (event) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!kpi || !guide) return null;

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="modal-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="kpi-modal-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <p className="modal-kicker">{storeName}</p>
            <h2 id="kpi-modal-title">{guide.title}</h2>
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className={`modal-snapshot status-${kpi.status}`}>
          <div>
            <span className="modal-snap-label">This store</span>
            <strong>{kpi.display}</strong>
          </div>
          <div>
            <span className="modal-snap-label">Network avg</span>
            <strong>{kpi.network_avg_display}</strong>
          </div>
          <div>
            <span className="modal-snap-label">Vs network</span>
            <strong>{vsLabel(kpi)}</strong>
          </div>
        </div>

        <section className="modal-section">
          <h3>Definition</h3>
          <p>{guide.meaning}</p>
        </section>

        <section className="modal-section">
          <h3>Business impact</h3>
          <p>{guide.whyItMatters}</p>
        </section>

        <section className="modal-section">
          <h3>Interpretation</h3>
          <ul>
            {guide.howToRead.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>

        <section className="modal-section">
          <h3>Recommended actions</h3>
          <ul>
            {guide.nextSteps.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}
