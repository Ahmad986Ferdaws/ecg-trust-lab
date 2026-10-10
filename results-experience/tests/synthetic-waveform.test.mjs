import assert from "node:assert/strict";
import test from "node:test";
import {
  SYNTHETIC_DURATION_SECONDS as duration,
  shiftedSyntheticEcg,
  syntheticEcg,
} from "../lib/synthetic-waveform.ts";

test("positive shifts zero-pad the start and preserve the retained strip for every lead", () => {
  for (let lead = 0; lead < 12; lead += 1) {
    for (let severity = 1; severity <= 4; severity += 1) {
      const shift = severity * 0.055;
      for (const time of [0, shift / 2, shift - 0.001]) {
        assert.equal(shiftedSyntheticEcg(time, lead, shift), 0);
      }
      for (const time of [shift, 0.5, 3, duration]) {
        assert.equal(shiftedSyntheticEcg(time, lead, shift), syntheticEcg(time - shift, lead));
      }
    }
  }
});

test("negative shifts zero-pad the end instead of wrapping another beat into view", () => {
  const shift = -0.22;
  for (let lead = 0; lead < 12; lead += 1) {
    assert.equal(shiftedSyntheticEcg(duration, lead, shift), 0);
    assert.equal(shiftedSyntheticEcg(duration - 0.1, lead, shift), 0);
    assert.equal(shiftedSyntheticEcg(0.16, lead, shift), syntheticEcg(0.38, lead));
  }
});

test("zero shift preserves every sampled value of the finite acquisition", () => {
  for (let lead = 0; lead < 12; lead += 1) {
    for (let sample = 0; sample < 960; sample += 1) {
      const time = (sample / 959) * duration;
      assert.equal(shiftedSyntheticEcg(time, lead, 0), syntheticEcg(time, lead));
    }
  }
});
