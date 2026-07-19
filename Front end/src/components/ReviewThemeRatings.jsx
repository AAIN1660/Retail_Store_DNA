export default function ReviewThemeRatings({ metrics = [], onSelect }) {
  return (
    <div className="review-theme-grid">
      {metrics.map((metric) => {
        const clickable = !metric.empty && typeof onSelect === "function";
        const Tag = clickable ? "button" : "article";
        return (
          <Tag
            key={metric.id}
            type={clickable ? "button" : undefined}
            className={`review-theme-card${metric.empty ? " is-empty" : ""}${clickable ? " is-clickable" : ""}`}
            onClick={clickable ? () => onSelect(metric) : undefined}
          >
            <p className="review-theme-label">{metric.label}</p>
            {metric.empty ? (
              <>
                <p className="review-theme-value empty">—</p>
                <p className="review-theme-meta">No customer comments on this</p>
              </>
            ) : (
              <>
                <p className="review-theme-value">{metric.display}</p>
                <p className="review-theme-meta">
                  Based on {metric.mention_count} review
                  {metric.mention_count === 1 ? "" : "s"}
                  {clickable ? " · Tap for summary" : ""}
                </p>
              </>
            )}
          </Tag>
        );
      })}
    </div>
  );
}
