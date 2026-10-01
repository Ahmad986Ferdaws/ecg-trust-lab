/** Finite acquisition used by the illustrative Failure Lab, never patient data. */
export const SYNTHETIC_DURATION_SECONDS = 6.2;

function gaussian(value: number, center: number, width: number): number {
  const distance = (value - center) / width;
  return Math.exp(-0.5 * distance * distance);
}

export function syntheticEcg(time: number, profileIndex: number): number {
  const phase = ((time % 1) + 1) % 1;
  const polarity = profileIndex === 3 || profileIndex === 6 ? -1 : 1;
  const scale = 0.74 + (profileIndex % 5) * 0.08;
  const p = 0.12 * gaussian(phase, 0.18, 0.035);
  const q = -0.2 * gaussian(phase, 0.355, 0.012);
  const r = 1.08 * gaussian(phase, 0.38, 0.014);
  const s = -0.32 * gaussian(phase, 0.415, 0.018);
  const t = 0.3 * gaussian(phase, 0.66, 0.07);
  return polarity * scale * (p + q + r + s + t);
}

/** Positive shifts delay the strip, matching the tracked Python corruption. */
export function shiftedSyntheticEcg(
  time: number,
  profileIndex: number,
  shiftSeconds: number,
): number {
  const sourceTime = time - shiftSeconds;
  if (sourceTime < 0 || sourceTime > SYNTHETIC_DURATION_SECONDS) return 0;
  return syntheticEcg(sourceTime, profileIndex);
}
