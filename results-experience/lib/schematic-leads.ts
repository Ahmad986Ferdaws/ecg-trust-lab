// Deterministic illustrative morphology; never a patient recording or model output.
export type LeadDefinition = {
  name: string;
  phase: number;
  polarity: number;
  p: number;
  r: number;
  s: number;
  t: number;
};

export const SCHEMATIC_LEADS: readonly LeadDefinition[] = [
  { name: "I", phase: 0.01, polarity: 1, p: 0.11, r: 0.78, s: 0.2, t: 0.27 },
  { name: "II", phase: 0, polarity: 1, p: 0.14, r: 1, s: 0.23, t: 0.34 },
  { name: "III", phase: 0.018, polarity: 1, p: 0.09, r: 0.58, s: 0.29, t: 0.23 },
  { name: "aVR", phase: 0.006, polarity: -1, p: 0.1, r: 0.72, s: 0.2, t: 0.26 },
  { name: "aVL", phase: 0.014, polarity: 1, p: 0.07, r: 0.43, s: 0.15, t: 0.18 },
  { name: "aVF", phase: 0.003, polarity: 1, p: 0.11, r: 0.76, s: 0.25, t: 0.28 },
  { name: "V1", phase: 0.012, polarity: 1, p: 0.07, r: 0.22, s: 0.72, t: -0.13 },
  { name: "V2", phase: 0.009, polarity: 1, p: 0.08, r: 0.42, s: 0.62, t: 0.2 },
  { name: "V3", phase: 0.005, polarity: 1, p: 0.1, r: 0.7, s: 0.44, t: 0.29 },
  { name: "V4", phase: 0, polarity: 1, p: 0.12, r: 1.02, s: 0.27, t: 0.35 },
  { name: "V5", phase: 0.004, polarity: 1, p: 0.11, r: 0.91, s: 0.19, t: 0.32 },
  { name: "V6", phase: 0.008, polarity: 1, p: 0.1, r: 0.7, s: 0.14, t: 0.27 },
] as const;

function gaussian(x: number, center: number, width: number, amplitude: number) {
  const distance = (x - center) / width;
  return amplitude * Math.exp(-0.5 * distance * distance);
}

export function sampledLeadValue(progress: number, lead: LeadDefinition) {
  const beatCount = 3.28;
  const phase = (progress * beatCount + lead.phase) % 1;
  const p = gaussian(phase, 0.18, 0.035, lead.p);
  const q = gaussian(phase, 0.365, 0.011, -lead.r * 0.13);
  const r = gaussian(phase, 0.395, 0.009, lead.r);
  const s = gaussian(phase, 0.43, 0.016, -lead.s);
  const t = gaussian(phase, 0.68, 0.07, lead.t);
  return lead.polarity * (p + q + r + s + t);
}


export function schematicLeadPath(lead: LeadDefinition, index: number) {
  const baseline = 175 + index * 58.2;
  return Array.from({ length: 520 }, (_, sample) => {
    const progress = sample / 519;
    const x = 70 + progress * 880;
    const y = baseline - sampledLeadValue(progress, lead) * 26;
    return `${sample === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
}
