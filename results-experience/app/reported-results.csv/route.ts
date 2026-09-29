import {
  AGGREGATE_RESULTS_FILENAME,
  aggregateResultsCsv,
} from "../../lib/aggregate-results-csv";

export function GET() {
  return new Response(aggregateResultsCsv(), {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": `attachment; filename="${AGGREGATE_RESULTS_FILENAME}"`,
      "X-Content-Type-Options": "nosniff",
    },
  });
}
