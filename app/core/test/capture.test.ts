/**
 * CaptureBuffer state machine, mirroring tests/test_capture.py case for case.
 * Frames are plain ints indexing a synthetic per-frame energy list, so idle
 * trimming never breaks the frame -> energy correspondence.
 */
import { describe, expect, it } from "vitest";
import { CaptureBuffer } from "../src/capture";

const energyFn = (perFrame: number[]) => (frames: number[]) => frames.map((i) => perFrame[i]!);
const zeros = (n: number) => Array<number>(n).fill(0);
const ones = (n: number) => Array<number>(n).fill(1);
const SENTENCE = [...zeros(5), ...ones(10), ...zeros(5), ...ones(10), ...zeros(5), ...ones(10), ...zeros(20)];

describe("CaptureBuffer", () => {
  it("stays idle while quiet, keeping only the preroll", () => {
    const buf = new CaptureBuffer(energyFn(zeros(20)), 0.5, 5, 3, 50);
    for (let i = 0; i < 10; i++) buf.append(i);
    expect(buf.state).toBe("idle");
    expect(buf.frames.length).toBeLessThanOrEqual(3);
  });

  it("goes active on a motion spike", () => {
    const buf = new CaptureBuffer(energyFn([...zeros(5), ...ones(20)]), 0.5, 5, 3, 50);
    for (let i = 0; i < 10; i++) buf.append(i);
    expect(buf.state).toBe("active");
  });

  it("settles only once the full settle window is quiet", () => {
    const buf = new CaptureBuffer(energyFn([...zeros(5), ...ones(5), ...zeros(20)]), 0.5, 4, 3, 100);
    for (let i = 0; i < 13; i++) buf.append(i);
    expect(buf.state).toBe("active");
    buf.append(13);
    expect(buf.state).toBe("settled");
  });

  it("forces a settle at the frame cap", () => {
    const buf = new CaptureBuffer(energyFn([...zeros(3), ...ones(100)]), 0.5, 5, 3, 20);
    for (let i = 0; i < 50 && buf.state !== "settled"; i++) buf.append(i);
    expect(buf.state).toBe("settled");
    expect(buf.frames.length).toBeLessThanOrEqual(20);
  });

  it("reset clears state", () => {
    const buf = new CaptureBuffer(energyFn(ones(10)), 0.5, 3, 3, 50);
    for (let i = 0; i < 6; i++) buf.append(i);
    buf.reset();
    expect(buf.state).toBe("idle");
    expect(buf.frames).toEqual([]);
  });

  it("is frozen once settled", () => {
    const buf = new CaptureBuffer(energyFn([...zeros(3), ...ones(5), ...zeros(10)]), 0.5, 4, 3, 50);
    for (let i = 0; i < 13; i++) buf.append(i);
    expect(buf.state).toBe("settled");
    const len = buf.frames.length;
    buf.append(999); // out of range: would break energyFn if not ignored
    expect(buf.frames.length).toBe(len);
  });

  it("sentence mode rides through brief inter-sign pauses", () => {
    const lastMotion = 44;
    const buf = new CaptureBuffer(energyFn(SENTENCE), 0.5, 15, 3, 200);
    for (let i = 0; i <= lastMotion + 14; i++) {
      buf.append(i);
      expect(buf.state, `frame ${i}`).not.toBe("settled");
    }
    buf.append(lastMotion + 15);
    expect(buf.state).toBe("settled");
  });

  it("a single-sign settle window ends a sentence mid-way", () => {
    const buf = new CaptureBuffer(energyFn(SENTENCE), 0.5, 4, 3, 200);
    for (let i = 0; i < 25 && buf.state !== "settled"; i++) buf.append(i);
    expect(buf.state).toBe("settled");
  });

  it("degrades to a capped trailing window if energyFn throws", () => {
    const buf = new CaptureBuffer<number>(() => {
      throw new Error("no hand blocks");
    }, 0.5, 4, 3, 10);
    for (let i = 0; i < 15; i++) buf.append(i);
    expect(buf.degraded).toBe(true);
    expect(buf.state).toBe("active");
    expect(buf.frames.length).toBeLessThanOrEqual(10);
  });
});
