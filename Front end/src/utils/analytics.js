export function filterStores(stores, { search, retailer, state }) {
  const q = search.trim().toLowerCase();
  return stores.filter((s) => {
    if (retailer && s.retailer !== retailer) return false;
    if (state && s.state !== state) return false;
    if (!q) return true;
    return (
      s.store_id.toLowerCase().includes(q) ||
      s.store_name.toLowerCase().includes(q) ||
      s.retailer.toLowerCase().includes(q) ||
      s.city.toLowerCase().includes(q) ||
      s.state.toLowerCase().includes(q)
    );
  });
}

export function uniqueStates(stores) {
  return [...new Set(stores.map((s) => s.state))].sort();
}
