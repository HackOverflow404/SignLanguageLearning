/**
 * Port of src/aslcv/capture.py's CaptureBuffer: motion-aware live capture
 * boundaries instead of a fixed trailing window (which would silently drop the
 * start of a slow attempt).
 *
 *   idle    -- no motion; buffer trimmed to `preroll` frames of rest lead-in
 *   active  -- motion seen; accumulate up to `maxFrames`
 *   settled -- `settleFrames` consecutive quiet frames (or the cap): a complete
 *              attempt, frozen until reset()
 *
 * `energyFn` is injected (handMotionEnergy bound to a skeleton, in the app) so
 * the state machine is testable with a synthetic signal. If it ever throws, the
 * buffer degrades to a plain capped trailing window instead of crashing the loop.
 */
export type CaptureState = "idle" | "active" | "settled";

export class CaptureBuffer<T> {
  frames: T[] = [];
  state: CaptureState = "idle";
  private energyOk = true;

  constructor(
    private readonly energyFn: (frames: T[]) => ArrayLike<number>,
    readonly motionThreshold: number,
    public settleFrames = 10,
    readonly preroll = 15,
    readonly maxFrames = 150,
  ) {}

  get degraded(): boolean {
    return !this.energyOk;
  }

  reset(): void {
    this.frames = [];
    this.state = "idle";
  }

  append(frame: T): void {
    if (this.state === "settled") return; // frozen until reset()
    this.frames.push(frame);

    if (!this.energyOk) {
      this.frames = this.frames.slice(-this.maxFrames);
      this.state = this.frames.length >= this.preroll ? "active" : "idle";
      return;
    }

    let energy: ArrayLike<number>;
    try {
      energy = this.energyFn(this.frames);
    } catch {
      this.energyOk = false;
      return;
    }
    const n = energy.length;
    if (n < 2) return; // not enough frames for a velocity-based signal yet

    if (this.state === "idle") {
      let moving = false;
      for (let i = Math.max(0, n - 3); i < n; i++) if (energy[i]! > this.motionThreshold) moving = true;
      if (moving) this.state = "active";
      else this.frames = this.frames.slice(-this.preroll);
      return;
    }

    // active
    if (this.frames.length >= this.maxFrames) {
      this.state = "settled";
      return;
    }
    if (n < this.settleFrames) return;
    for (let i = n - this.settleFrames; i < n; i++) if (energy[i]! > this.motionThreshold) return;
    this.state = "settled";
  }
}
