export default function SimpleKpiCard({ kpi, onSelect }) {
  const vsText =
    kpi.vs_network === "on par"
      ? "Same as network avg"
      : kpi.vs_network === "above"
        ? "Above network avg"
        : kpi.vs_network === "below"
          ? "Below network avg"
          : "Above network avg";

  return (
    <button
      type="button"
      className={`kpi-card simple-kpi status-${kpi.status}`}
      onClick={() => onSelect?.(kpi)}
    >
      <span className="kpi-label">{kpi.label}</span>
      <span className="kpi-value">{kpi.display}</span>
      <span className="kpi-benchmark">Network avg: {kpi.network_avg_display}</span>
      <span className="kpi-vs">{vsText}</span>
    </button>
  );
}
