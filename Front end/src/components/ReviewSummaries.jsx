export default function ReviewSummaries({ summaries }) {
  const positive = summaries?.positive ?? { count: 0, summary: "", excerpts: [] };
  const negative = summaries?.negative ?? { count: 0, summary: "", excerpts: [] };

  return (
    <div className="review-summary-grid">
      <article className="review-summary-card positive">
        <header>
          <h4>Positive review summary</h4>
          <span>{positive.count} review{positive.count === 1 ? "" : "s"}</span>
        </header>
        {positive.summary ? (
          <p>{positive.summary}</p>
        ) : (
          <p className="empty-copy">No positive reviews for this store yet.</p>
        )}
      </article>
      <article className="review-summary-card negative">
        <header>
          <h4>Negative review summary</h4>
          <span>{negative.count} review{negative.count === 1 ? "" : "s"}</span>
        </header>
        {negative.summary ? (
          <p>{negative.summary}</p>
        ) : (
          <p className="empty-copy">No negative reviews for this store yet.</p>
        )}
      </article>
    </div>
  );
}
