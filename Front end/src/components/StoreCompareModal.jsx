import { useEffect, useMemo, useState } from "react";
import { fetchCompare } from "../api/client";

function StatusBadge({ status, tone }) {
  return <span className={`status-badge tone-${tone}`}>{status}</span>;
}

function kpiWinner(kpiA, kpiB) {
  if (!kpiB || kpiA.value === kpiB.value) return "tie";
  const aWins = kpiA.higher_is_better ? kpiA.value > kpiB.value : kpiA.value < kpiB.value;
  return aWins ? "a" : "b";
}

function formatDelta(kpiA, kpiB) {
  if (!kpiB) return "—";
  const delta = kpiB.value - kpiA.value;
  if (kpiA.id === "complaints") return `${delta > 0 ? "+" : ""}${Math.round(delta)}`;
  return `${delta > 0 ? "+" : ""}${(delta * 100).toFixed(1)} pts`;
}

function shortPlace(store) {
  return store.city || store.store_name.split(" ").slice(-1)[0];
}

export default function StoreCompareModal({ storeA, storeB, onClose }) {
  const [matchPct, setMatchPct] = useState(storeA?.similar_to?.[storeB?.store_id] ?? null);
  const [peerLearnings, setPeerLearnings] = useState(null);
  const [learningsStatus, setLearningsStatus] = useState("idle");
  const [showSharedLearnings, setShowSharedLearnings] = useState(false);

  useEffect(() => {
    const onKey = (event) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const differentCategory = useMemo(() => {
    if (!storeA || !storeB) return false;
    return storeA.executive.status_tone !== storeB.executive.status_tone;
  }, [storeA, storeB]);

  useEffect(() => {
    if (!storeA || !storeB) return;
    let cancelled = false;
    setLearningsStatus(differentCategory ? "loading" : "idle");
    setPeerLearnings(null);
    setShowSharedLearnings(false);

    fetchCompare(storeA.store_id, storeB.store_id)
      .then((result) => {
        if (cancelled) return;
        if (result.similarity != null) setMatchPct(result.similarity);
        else if (storeA.similar_to?.[storeB.store_id] != null) {
          setMatchPct(storeA.similar_to[storeB.store_id]);
        }
        setPeerLearnings(result.peer_learnings ?? null);
        setLearningsStatus(result.peer_learnings ? "ready" : "none");
      })
      .catch(() => {
        if (cancelled) return;
        const local = storeA.similar_to?.[storeB.store_id];
        if (local != null) setMatchPct(local);
        setPeerLearnings(null);
        setLearningsStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, [storeA, storeB, differentCategory]);

  const learningByKpi = useMemo(() => {
    const map = {};
    for (const item of peerLearnings?.learnings || []) {
      map[item.kpi_id] = item;
    }
    return map;
  }, [peerLearnings]);

  const rows = useMemo(() => {
    if (!storeA || !storeB) return [];
    return storeA.executive.kpis.map((kpiA) => {
      const kpiB = storeB.executive.kpis.find((k) => k.id === kpiA.id);
      const winner = kpiWinner(kpiA, kpiB);
      return {
        id: kpiA.id,
        label: kpiA.label,
        valueA: kpiA.display,
        valueB: kpiB?.display ?? "—",
        winner,
        delta: formatDelta(kpiA, kpiB),
      };
    });
  }, [storeA, storeB]);

  const tipCards = useMemo(() => {
    if (!storeA || !storeB || !differentCategory) return [];
    return storeA.executive.kpis
      .map((kpiA) => {
        const kpiB = storeB.executive.kpis.find((k) => k.id === kpiA.id);
        const winner = kpiWinner(kpiA, kpiB);
        if (winner === "tie") return null;
        const learning = learningByKpi[kpiA.id];
        if (!learning && learningsStatus !== "loading") return null;
        const leader = winner === "a" ? storeA : storeB;
        const tips = [];
        const headline = learning?.headline?.trim();
        if (headline) tips.push(headline);
        for (const bullet of learning?.bullets || []) {
          const text = String(bullet || "").trim();
          if (text && !tips.includes(text)) tips.push(text);
          if (tips.length >= 3) break;
        }
        if (!tips.length && learning?.summary) tips.push(String(learning.summary).trim());
        if (!tips.length) tips.push(`Copy practices from ${shortPlace(leader)}`);
        return {
          id: kpiA.id,
          label: kpiA.label,
          tips: tips.slice(0, 3),
          loading: !learning && learningsStatus === "loading",
        };
      })
      .filter(Boolean);
  }, [storeA, storeB, differentCategory, learningByKpi, learningsStatus]);

  if (!storeA || !storeB) return null;

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="modal-panel compare-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="compare-modal-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <p className="modal-kicker">Store comparison</p>
            <h2 id="compare-modal-title">Head-to-head benchmark</h2>
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="compare-modal-score">
          <span className="compare-modal-score-value">
            {matchPct != null ? `${Math.round(matchPct * 100)}%` : "—"}
          </span>
          <span className="compare-modal-score-label">Playbook match</span>
        </div>

        <div className="compare-modal-sides">
          <section className="compare-modal-side">
            <h3>{storeA.store_name}</h3>
            <p className="subtle">
              {storeA.retailer} · {storeA.city}, {storeA.state}
            </p>
            <div className="compare-modal-status">
              <StatusBadge
                status={storeA.executive.status}
                tone={storeA.executive.status_tone}
              />
              <span>
                {storeA.executive.metrics_above_avg} of 5 metrics at or above network avg
              </span>
            </div>
          </section>
          <section className="compare-modal-side">
            <h3>{storeB.store_name}</h3>
            <p className="subtle">
              {storeB.retailer} · {storeB.city}, {storeB.state}
            </p>
            <div className="compare-modal-status">
              <StatusBadge
                status={storeB.executive.status}
                tone={storeB.executive.status_tone}
              />
              <span>
                {storeB.executive.metrics_above_avg} of 5 metrics at or above network avg
              </span>
            </div>
          </section>
        </div>

        <section className="modal-section">
          <h3>Metric comparison</h3>
          <p className="card-desc">Green highlights who is ahead on each KPI.</p>
          <div className="compare-table-wrap">
            <table className="data-table benchmark-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>{storeA.store_name}</th>
                  <th>{storeB.store_name}</th>
                  <th>Gap</th>
                  <th>Ahead</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.id}>
                    <td>
                      <strong>{row.label}</strong>
                    </td>
                    <td className={row.winner === "a" ? "winner-cell" : ""}>{row.valueA}</td>
                    <td className={row.winner === "b" ? "winner-cell" : ""}>{row.valueB}</td>
                    <td>{row.delta}</td>
                    <td>
                      {row.winner === "tie"
                        ? "Tied"
                        : row.winner === "a"
                          ? shortPlace(storeA)
                          : shortPlace(storeB)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {differentCategory && (
          <section className="modal-section playbook-section">
            <div className="playbook-section-head">
              <div>
                <h3>Shared learnings</h3>
                <p className="card-desc">
                  Peer tips
                  {learningsStatus === "error" ? ". Tips could not be loaded." : "."}
                </p>
              </div>
              <button
                type="button"
                className={`playbook-toggle${showSharedLearnings ? " is-open" : ""}`}
                onClick={() => setShowSharedLearnings((open) => !open)}
                aria-expanded={showSharedLearnings}
              >
                Shared learning
              </button>
            </div>

            {showSharedLearnings && (
              <>
                {tipCards.length === 0 && learningsStatus === "loading" ? (
                  <p className="muted playbook-empty">Drafting shared learnings…</p>
                ) : null}

                {tipCards.length === 0 && learningsStatus !== "loading" ? (
                  <p className="muted playbook-empty">No clear shared learnings for this pair.</p>
                ) : null}

                <div className="playbook-grid">
                  {tipCards.map((tip) => (
                    <article key={tip.id} className="playbook-card">
                      <h4 className="playbook-metric">{tip.label}</h4>
                      {tip.loading ? (
                        <p className="muted playbook-loading">Drafting tip…</p>
                      ) : (
                        <ol className="playbook-tips">
                          {tip.tips.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ol>
                      )}
                    </article>
                  ))}
                </div>
              </>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
