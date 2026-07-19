export default function SentimentBreakdown({ voice }) {
  const total = voice.review_count || 1;
  const segments = [
    { key: "positive", label: "Pos", count: voice.positive, color: "#22c55e" },
    { key: "neutral", label: "Neu", count: voice.neutral, color: "#94a3b8" },
    { key: "negative", label: "Neg", count: voice.negative, color: "#ec4899" },
  ];

  return (
    <div className="sentiment-block">
      <ul className="sentiment-legend" aria-label="Sentiment legend">
        {segments.map((seg) => (
          <li key={seg.key}>
            <span className="legend-dot" style={{ background: seg.color }} />
            <span className="legend-label">{seg.label}</span>
            <strong>{seg.count}</strong>
          </li>
        ))}
      </ul>
      <div className="sentiment-bar" aria-hidden="true">
        {segments.map((seg) => (
          <div
            key={seg.key}
            className="sentiment-segment"
            style={{
              width: `${(seg.count / total) * 100}%`,
              background: seg.color,
            }}
            title={`${seg.label}: ${seg.count}`}
          />
        ))}
      </div>
      <p className="sentiment-summary">
        {voice.review_count} reviews · {voice.positive_share_display} positive ·{" "}
        {voice.negative_share_display} negative
      </p>
    </div>
  );
}
