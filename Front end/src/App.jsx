import { useEffect, useState } from "react";
import { fetchPortfolio } from "./api/client";
import PortfolioView from "./views/PortfolioView";
import StoreExplorerView from "./views/StoreExplorerView";

const TABS = [
  { id: "portfolio", label: "Executive Summary", hint: "Portfolio KPIs" },
  { id: "explorer", label: "Store Pulse", hint: "Reviews & news" },
];

export default function App() {
  const [tab, setTab] = useState("portfolio");
  const [summary, setSummary] = useState(null);
  const [stores, setStores] = useState([]);
  const [selectedStoreId, setSelectedStoreId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchPortfolio()
      .then((data) => {
        if (cancelled) return;
        setSummary(data.summary);
        setStores(data.stores);
        setSelectedStoreId(data.stores[0]?.store_id ?? null);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message || "Failed to load stores");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <div className="dashboard-root">
        <div className="app-state center">
          <div className="loader" />
          <p>Loading dashboard…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-root">
        <div className="app-state center">
          <h2>Could not load store data</h2>
          <p className="muted">{error}</p>
          <p className="muted">
            Ensure curated CSVs exist under <code>data/USA_100_Stores/</code>, Stage 6 store
            embeddings are indexed, then start the API:
            <code>python -m uvicorn api.main:app --reload --port 8000</code>
          </p>
        </div>
      </div>
    );
  }

  if (!stores.length || !summary) {
    return (
      <div className="dashboard-root">
        <div className="app-state center">
          <p className="muted">No stores available yet.</p>
        </div>
      </div>
    );
  }

  const pageTitle = TABS.find((t) => t.id === tab)?.label ?? "Dashboard";

  return (
    <div className="dashboard-root">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="brand-mark" aria-hidden="true">
            ◆
          </span>
          <div>
            <strong>Store DNA</strong>
            <span>Retail analytics</span>
          </div>
        </div>

        <p className="sidebar-section">Main menu</p>
        <nav className="sidebar-nav">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              className={tab === t.id ? "nav-item active" : "nav-item"}
              onClick={() => setTab(t.id)}
            >
              <span className="nav-dot" aria-hidden="true" />
              <span>
                <strong>{t.label}</strong>
                <small>{t.hint}</small>
              </span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <p>Need help?</p>
          <span>Use the store list to switch locations and review KPIs.</span>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <div>
            <p className="topbar-kicker">Dashboard</p>
            <h1>{pageTitle}</h1>
          </div>
          <div className="topbar-meta">
            <span className="meta-chip">{summary.store_count} stores</span>
            <span className="meta-chip soft">{summary.retailer_count} retailers</span>
          </div>
        </header>

        <main className="workspace-main" key={tab}>
          {tab === "portfolio" && (
            <PortfolioView
              summary={summary}
              stores={stores}
              selectedStoreId={selectedStoreId}
              onSelectStore={setSelectedStoreId}
            />
          )}
          {tab === "explorer" && (
            <StoreExplorerView
              stores={stores}
              selectedStoreId={selectedStoreId}
              onSelectStore={setSelectedStoreId}
            />
          )}
        </main>
      </div>
    </div>
  );
}
