const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function request(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

export function fetchPortfolio() {
  return request("/api/stores");
}

export function fetchCompare(storeA, storeB) {
  const params = new URLSearchParams({ store_a: storeA, store_b: storeB });
  return request(`/api/compare?${params.toString()}`);
}
