import { COHORT_IDS, type CohortId } from "./results.ts";

export function cohortFromSearch(search: string): CohortId | null {
  const value = new URLSearchParams(search).get("cohort");
  return COHORT_IDS.find((id) => id === value) ?? null;
}

export function urlForCohort(currentUrl: string, cohort: CohortId): string {
  const url = new URL(currentUrl);
  url.searchParams.set("cohort", cohort);
  return `${url.pathname}${url.search}${url.hash}`;
}
