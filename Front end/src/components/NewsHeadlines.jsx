import { useEffect, useState } from "react";

export default function NewsHeadlines({ news }) {
  const headlines = news?.headlines ?? [];
  const [openIndex, setOpenIndex] = useState(null);

  useEffect(() => {
    setOpenIndex(null);
  }, [news]);

  if (!headlines.length) {
    return <p className="empty-copy">No local news headlines for this store yet.</p>;
  }

  return (
    <div className="news-headlines">
      <ul className="news-headline-list">
        {headlines.map((item, index) => {
          const isOpen = openIndex === index;
          return (
            <li key={`${item.headline}-${index}`}>
              <button
                type="button"
                className={`news-headline-card${isOpen ? " is-open" : ""}`}
                onClick={() => setOpenIndex(isOpen ? null : index)}
                aria-expanded={isOpen}
              >
                <div className="news-headline-top">
                  <div className="news-headline-title-row">
                    <h4>{item.headline}</h4>
                    <span className="news-expand-hint" aria-hidden="true">
                      {isOpen ? "−" : "+"}
                    </span>
                  </div>
                  {item.published_date ? (
                    <div className="news-headline-meta">
                      <span>{item.published_date}</span>
                    </div>
                  ) : null}
                </div>

                {isOpen ? (
                  <div className="news-headline-detail">
                    {(item.event_type || item.demand_impact) && (
                      <div className="news-headline-meta">
                        {item.event_type ? (
                          <span className="news-tag">
                            {item.event_type.replaceAll("_", " ")}
                          </span>
                        ) : null}
                        {item.demand_impact ? (
                          <span className={`news-tag impact-${item.demand_impact}`}>
                            {item.demand_impact}
                          </span>
                        ) : null}
                      </div>
                    )}
                    {item.summary ? <p>{item.summary}</p> : null}
                    {item.why_it_matters ? (
                      <p className="news-why">
                        <strong>Why it matters:</strong> {item.why_it_matters}
                      </p>
                    ) : null}
                  </div>
                ) : null}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
