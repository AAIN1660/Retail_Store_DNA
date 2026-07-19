export const KPI_GUIDE = {
  fulfillment: {
    title: "Fulfillment rate",
    meaning:
      "The share of customer orders this store completed on time and in full. A higher rate means shoppers got what they ordered without delays or missing items.",
    whyItMatters:
      "Missed fulfillments drive cancellations, refunds, and weaker guest trust — especially for pickup and delivery.",
    howToRead: [
      "Higher is better.",
      "Compare against the network average to see if this store is keeping pace.",
      "A small gap can still matter when order volume is high.",
    ],
    watchFor: "Sustained rates below network average, or sharp week-to-week drops during busy periods.",
    nextSteps: [
      "Check pick rates and staffing during peak hours.",
      "Review top items that fail fulfillment most often.",
      "Confirm inventory accuracy for high-demand SKUs.",
    ],
  },
  oos: {
    title: "Out-of-stock rate",
    meaning:
      "The share of tracked items that were unavailable when customers wanted them. Lower means shelves and systems are better aligned with demand.",
    whyItMatters:
      "Empty shelves lose sales immediately and push shoppers to competitors or substitutes.",
    howToRead: [
      "Lower is better.",
      "Above network average signals replenishment or forecasting pressure.",
      "Even a few points can hit high-velocity categories hard.",
    ],
    watchFor: "Rising OOS in key categories, or gaps that repeat on the same SKUs week after week.",
    nextSteps: [
      "Review top out-of-stock items and replenishment timing.",
      "Validate on-hand counts vs. system inventory.",
      "Coordinate with supply for recurring shortages.",
    ],
  },
  shrink: {
    title: "Shrink rate",
    meaning:
      "Inventory loss as a share of sales — from theft, damage, spoilage, or process errors. Lower shrink protects margin.",
    whyItMatters:
      "Shrink quietly erodes profit. Small percentage points add up across a full store assortment.",
    howToRead: [
      "Lower is better.",
      "At or above network average deserves attention.",
      "Look at both the rate and which categories drive it.",
    ],
    watchFor: "Sudden spikes, high-shrink departments (for example fresh or high-theft zones), or process gaps after deliveries.",
    nextSteps: [
      "Walk high-risk areas with the store team.",
      "Tighten receiving and cycle-count routines.",
      "Review damage and spoilage logs for patterns.",
    ],
  },
  sentiment: {
    title: "Positive reviews",
    meaning:
      "The share of customer reviews that read as positive. It is a simple pulse on how guests feel about the store experience.",
    whyItMatters:
      "Review tone influences reputation, foot traffic, and whether issues are being recovered well.",
    howToRead: [
      "Higher is better.",
      "Above network average is a strength to protect.",
      "Pair this with complaint volume for a fuller guest picture.",
    ],
    watchFor: "Falling positive share, or positive volume that is high while complaints are also climbing.",
    nextSteps: [
      "Read recent negative themes with the team.",
      "Celebrate and replicate what guests praise.",
      "Close the loop on recurring service complaints.",
    ],
  },
  complaints: {
    title: "Customer complaints",
    meaning:
      "How many customer complaint issues were logged for this store in the recent window. Fewer complaints usually means fewer unresolved pain points.",
    whyItMatters:
      "Complaints are early warnings — service, stock, cleanliness, or checkout friction often shows up here first.",
    howToRead: [
      "Lower is better.",
      "Below network average is a good signal.",
      "A low count with weak review sentiment may still hide quieter dissatisfaction.",
    ],
    watchFor: "Clusters of the same complaint type, or a jump after schedule or assortment changes.",
    nextSteps: [
      "Group complaints by theme and assign owners.",
      "Fix the top two recurring issues first.",
      "Follow up on open cases until they are closed.",
    ],
  },
};

export function getKpiGuide(kpiId) {
  return KPI_GUIDE[kpiId] ?? null;
}
