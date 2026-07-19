import { useMemo, useState } from "react";
import ReviewThemeRatings from "../components/ReviewThemeRatings";
import ReviewSummaries from "../components/ReviewSummaries";
import NewsHeadlines from "../components/NewsHeadlines";
import ThemeSummaryModal from "../components/ThemeSummaryModal";
import { filterStores, uniqueStates } from "../utils/analytics";

export default function StoreExplorerView({ stores, selectedStoreId, onSelectStore }) {
  const [search, setSearch] = useState("");
  const [retailer, setRetailer] = useState("");
  const [state, setState] = useState("");
  const [selectedTheme, setSelectedTheme] = useState(null);

  const filtered = useMemo(
    () => filterStores(stores, { search, retailer, state }),
    [stores, search, retailer, state],
  );

  const selected = useMemo(
    () => stores.find((s) => s.store_id === selectedStoreId) ?? stores[0],
    [stores, selectedStoreId],
  );

  const retailers = useMemo(
    () => [...new Set(stores.map((s) => s.retailer))].sort(),
    [stores],
  );
  const states = useMemo(() => uniqueStates(stores), [stores]);
  const pulse = selected.business.store_pulse;
  const themeRatings = pulse.reviews.theme_ratings ?? [];
  const reviewSummaries = pulse.reviews.summaries ?? null;

  return (
    <div className="view explorer-view">
      <aside className="explorer-list card">
        <h3>Select a store</h3>
        <input
          className="search"
          type="search"
          placeholder="Search stores..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <div className="filter-row">
          <select value={retailer} onChange={(e) => setRetailer(e.target.value)}>
            <option value="">All retailers</option>
            {retailers.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
          <select value={state} onChange={(e) => setState(e.target.value)}>
            <option value="">All states</option>
            {states.map((st) => (
              <option key={st} value={st}>
                {st}
              </option>
            ))}
          </select>
        </div>
        <ul className="store-list">
          {filtered.map((store) => (
            <li key={store.store_id}>
              <button
                type="button"
                className={
                  store.store_id === selected.store_id ? "store-item active" : "store-item"
                }
                onClick={() => onSelectStore(store.store_id)}
              >
                <strong>{store.store_name}</strong>
                <span>
                  {store.business.store_pulse.reviews.breakdown.review_count} reviews ·{" "}
                  {store.business.store_pulse.news.article_count ??
                    store.business.store_pulse.news.headlines?.length ??
                    0}{" "}
                  news · {store.city}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main className="explorer-detail">
        <section className="card detail-hero">
          <p className="eyebrow dark">Store pulse</p>
          <h2>{selected.store_name}</h2>
          <p className="subtle">
            {selected.retailer} · {selected.format} · {selected.city}, {selected.state}
          </p>
        </section>

        <section className="card detail-section">
          <h3>Customer reviews</h3>
          <ReviewThemeRatings metrics={themeRatings} onSelect={setSelectedTheme} />

          <h4 className="pulse-subhead">Review summaries</h4>
          <ReviewSummaries summaries={reviewSummaries} />
        </section>

        <section className="card detail-section">
          <h3>Local news headlines</h3>
          <NewsHeadlines news={pulse.news} />
        </section>
      </main>

      {selectedTheme && (
        <ThemeSummaryModal
          metric={selectedTheme}
          storeName={selected.store_name}
          onClose={() => setSelectedTheme(null)}
        />
      )}
    </div>
  );
}
