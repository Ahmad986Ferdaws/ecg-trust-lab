import assert from "node:assert/strict";
import test from "node:test";
import {
  SCHEMATIC_LEADS,
  sampledLeadValue,
  schematicLeadPath,
} from "../lib/schematic-leads.ts";

test("the schematic preserves all twelve named, non-flat lead morphologies", () => {
  assert.deepEqual(SCHEMATIC_LEADS.map((lead) => lead.name), [
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
  ]);

  for (const [index, lead] of SCHEMATIC_LEADS.entries()) {
    const path = schematicLeadPath(lead, index);
    assert.equal(path, schematicLeadPath(lead, index), `${lead.name} is deterministic`);
    const points = [...path.matchAll(/[ML]([\d.]+),([\d.]+)/g)]
      .map(([, x, y]) => [Number(x), Number(y)]);
    assert.equal(points.length, 520);
    assert.equal(points[0][0], 70);
    assert.equal(points.at(-1)[0], 950);

    const ys = points.map(([, y]) => y);
    assert.ok(Math.max(...ys) - Math.min(...ys) > 10, `${lead.name} must not be a flat line`);
    assert.ok(ys.every((y) => y > 145 + index * 58.2 && y < 205 + index * 58.2),
      `${lead.name} remains in its own visible strip`);
    assert.ok(points.slice(1).every(([x], i) => x > points[i][0]),
      `${lead.name} follows increasing time`);
  }
});

test("the shared waveform has the expected upright and inverted QRS complexes", () => {
  const leadII = SCHEMATIC_LEADS.find((lead) => lead.name === "II");
  const leadAVR = SCHEMATIC_LEADS.find((lead) => lead.name === "aVR");
  assert.ok(sampledLeadValue(0.395 / 3.28, leadII) > 0.95);
  assert.ok(sampledLeadValue((0.395 - leadAVR.phase) / 3.28, leadAVR) < -0.68);
  assert.ok(Math.abs(sampledLeadValue(0, leadII)) < 0.001);
});
