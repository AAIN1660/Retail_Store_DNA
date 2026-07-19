export default function PulseKpiCard({ kpi }) {
  return (
    <article className={`pulse-kpi-card status-${kpi.status}`}>
      <span className="kpi-label">{kpi.label}</span>
      <span className="kpi-value">{kpi.display}</span>
      <span className="kpi-benchmark">Network avg: {kpi.network_avg_display}</span>
    </article>
  );
}
