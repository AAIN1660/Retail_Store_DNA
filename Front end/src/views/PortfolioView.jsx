import { useMemo, useState } from "react";
import KpiDetailModal from "../components/KpiDetailModal";
import SimpleKpiCard from "../components/SimpleKpiCard";
import StoreCompareModal from "../components/StoreCompareModal";
import { filterStores, uniqueStates } from "../utils/analytics";

function StatusBadge({ status, tone }) {
  return <span className={`status-badge tone-${tone}`}>{status}</span>;
}

export default function PortfolioView({ summary, stores, selectedStoreId, onSelectStore }) {
  const [search, setSearch] = useState("");
  const [retailer, setRetailer] = useState("");
  const [state, setState] = useState("");
  const [selectedKpi, setSelectedKpi] = useState(null);
  const [comparePeerId, setComparePeerId] = useState(null);

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
  const comparePeer = useMemo(
    () => (comparePeerId ? stores.find((s) => s.store_id === comparePeerId) : null),
    [stores, comparePeerId],
  );
  const { executive } = selected;
  const rollups = summary.status_rollups;
  const totalStores = summary.store_count || 1;
  const statusCards = [
    {
      key: "good",
      label: "Doing well",
      count: rollups["Doing well"] ?? 0,
      tone: "good",
    },
    {
      key: "steady",
      label: "Steady",
      count: rollups.Steady ?? 0,
      tone: "stable",
    },
    {
      key: "focus",
      label: "Needs focus",
      count: rollups["Needs focus"] ?? 0,
      tone: "attention",
    },
  ];

  return (
    <div className="view executive-view">
      <section className="network-pulse">
        <div className="pulse-stats">
          {statusCards.map((card) => {
            const share = Math.round((card.count / totalStores) * 100);
            return (
              <div key={card.key} className={`pulse-item ${card.tone}`}>
                <div className="pulse-icon" aria-hidden="true" />
                <div className="pulse-body">
                  <div className="pulse-head">
                    <span>{card.label}</span>
                    <strong>{card.count}</strong>
                  </div>
                  <div className="pulse-meter" aria-hidden="true">
                    <div className="pulse-meter-fill" style={{ width: `${share}%` }} />
                  </div>
                  <p className="pulse-share">{share}% of portfolio</p>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <div className="executive-layout">
        <aside className="card executive-picker">
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
          <ul className="store-list compact">
            {filtered.map((store) => (
              <li key={store.store_id}>
                <button
                  type="button"
                  className={
                    store.store_id === selected.store_id ? "store-item active" : "store-item"
                  }
                  onClick={() => onSelectStore(store.store_id)}
                >
                  <div className="store-item-top">
                    <strong>{store.store_name}</strong>
                    <StatusBadge
                      status={store.executive.status}
                      tone={store.executive.status_tone}
                    />
                  </div>
                  <span>
                    {store.executive.metrics_above_avg}/5 vs avg · {store.retailer}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <main className="executive-main">
          <section className={`card status-hero tone-${executive.status_tone}`}>
            <div className="status-hero-top">
              <StatusBadge status={executive.status} tone={executive.status_tone} />
            </div>
            <h2>{selected.store_name}</h2>
            <p className="subtle">
              {selected.retailer} · {selected.format} · {selected.city}, {selected.state}
            </p>
            <p className="status-headline">{executive.headline}</p>
          </section>

          <section className="kpi-section-header">
            <h3>Key metrics</h3>
            <p className="card-desc">Compared to the network average.</p>
          </section>
          <section className="kpi-grid">
            {executive.kpis.map((kpi) => (
              <SimpleKpiCard key={kpi.id} kpi={kpi} onSelect={setSelectedKpi} />
            ))}
          </section>

          <section className="card">
            <h3>Similar stores</h3>
            <p className="card-desc">Stores with a similar operating pattern.</p>
            <div className="similar-stores">
              {selected.top_peers.slice(0, 3).map((peer) => (
                <button
                  type="button"
                  key={peer.store_id}
                  className="similar-store-card"
                  onClick={() => setComparePeerId(peer.store_id)}
                >
                  <strong>{peer.store_name}</strong>
                  <span>
                    {peer.retailer} · {peer.city}, {peer.state}
                  </span>
                  <span className="similarity-tag">
                    {Math.round(peer.similarity * 100)}% similar
                  </span>
                </button>
              ))}
            </div>
          </section>
        </main>
      </div>

      {selectedKpi && (
        <KpiDetailModal
          kpi={selectedKpi}
          storeName={selected.store_name}
          onClose={() => setSelectedKpi(null)}
        />
      )}

      {comparePeer && (
        <StoreCompareModal
          storeA={selected}
          storeB={comparePeer}
          onClose={() => setComparePeerId(null)}
        />
      )}
    </div>
  );
}
