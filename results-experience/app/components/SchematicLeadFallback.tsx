import { useId } from "react";
import { SCHEMATIC_LEADS, schematicLeadPath } from "../../lib/schematic-leads";

const paths = SCHEMATIC_LEADS.map((lead, index) => ({
  name: lead.name,
  d: schematicLeadPath(lead, index),
}));

export function SchematicLeadFallback() {
  const gradientId = useId();

  return (
    <svg
      aria-hidden="true"
      focusable="false"
      data-schematic-fallback=""
      viewBox="0 0 1000 1000"
      preserveAspectRatio="none"
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        backgroundColor: "#EEE7D8",
        backgroundImage:
          "repeating-linear-gradient(0deg, transparent 0 15px, rgba(152,72,59,.09) 15px 16px), repeating-linear-gradient(90deg, transparent 0 15px, rgba(152,72,59,.09) 15px 16px)",
      }}
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="0">
          <stop offset="50%" stopColor="#006B70" />
          <stop offset="50%" stopColor="#98483B" />
        </linearGradient>
      </defs>
      {paths.map(({ name, d }) => (
        <path
          key={name}
          data-schematic-lead={name}
          d={d}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeWidth={1.35}
          strokeOpacity={0.82}
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  );
}
