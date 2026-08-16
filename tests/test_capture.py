"""CaptureBuffer: the idle -> active -> settled state machine, with a
synthetic `energy_fn` (real per-frame lookup by frame VALUE, not a call
counter) so trimming during idle doesn't break the correspondence between
"which real frame is this" and "what energy does it carry" -- exercises the
settle-window logic faithfully, no real pose/MediaPipe fixtures needed.

Runs under pytest OR as a plain script (`python tests/test_capture.py`).
"""
import sys

from aslcv.capture import CaptureBuffer


def _energy_fn(per_frame_energy):
    """frames are plain ints indexing into `per_frame_energy` -- so `frames`
    (even after CaptureBuffer trims it during idle) always maps to the REAL
    per-frame energies at those positions, exactly like a real energy
    function fed the corresponding real poses would."""
    def fn(frames):
        return [per_frame_energy[f] for f in frames]
    return fn


def test_stays_idle_while_quiet():
    energies = [0.0] * 20
    buf = CaptureBuffer(_energy_fn(energies), motion_threshold=0.5, settle_frames=5, preroll=3, max_frames=50)
    for i in range(10):
        buf.append(i)
    assert buf.state == "idle"
    assert len(buf.frames) <= 3


def test_transitions_to_active_on_motion_spike():
    energies = [0.0] * 5 + [1.0] * 20
    buf = CaptureBuffer(_energy_fn(energies), motion_threshold=0.5, settle_frames=5, preroll=3, max_frames=50)
    for i in range(10):
        buf.append(i)
    assert buf.state == "active"


def test_settles_only_after_full_settle_window_is_quiet():
    # motion at frames 5-9, quiet from frame 10 onward
    energies = [0.0] * 5 + [1.0] * 5 + [0.0] * 20
    buf = CaptureBuffer(_energy_fn(energies), motion_threshold=0.5, settle_frames=4, preroll=3, max_frames=100)
    for i in range(13):  # only 3 consecutive quiet frames follow the last moving one so far
        buf.append(i)
    assert buf.state == "active"
    buf.append(13)  # 4th consecutive quiet frame -- now the full settle window is quiet
    assert buf.state == "settled"


def test_max_frames_cap_forces_settle_even_without_a_pause():
    energies = [0.0] * 3 + [1.0] * 100  # sustained motion, never stops on its own
    buf = CaptureBuffer(_energy_fn(energies), motion_threshold=0.5, settle_frames=5, preroll=3, max_frames=20)
    for i in range(50):
        buf.append(i)
        if buf.state == "settled":
            break
    assert buf.state == "settled"
    assert len(buf.frames) <= 20


def test_reset_clears_state():
    energies = [1.0] * 10
    buf = CaptureBuffer(_energy_fn(energies), motion_threshold=0.5, settle_frames=3, preroll=3, max_frames=50)
    for i in range(6):
        buf.append(i)
    buf.reset()
    assert buf.state == "idle"
    assert buf.frames == []


def test_settled_buffer_frozen_until_reset():
    energies = [0.0] * 3 + [1.0] * 5 + [0.0] * 10
    buf = CaptureBuffer(_energy_fn(energies), motion_threshold=0.5, settle_frames=4, preroll=3, max_frames=50)
    for i in range(13):
        buf.append(i)
    assert buf.state == "settled"
    frozen_len = len(buf.frames)
    buf.append(999)  # frame value out of range -- would KeyError/IndexError if not ignored
    assert len(buf.frames) == frozen_len


def test_sentence_mode_survives_brief_inter_sign_pauses():
    """Phase 7 step 5: a multi-sign sentence attempt has SEVERAL motion
    rises, with brief dips between words rather than one clean rise-and-fall.
    A `settle_frames` tuned for single-sign use (short) would end the capture
    at the first inter-word pause; a larger `settle_frames` (sentence-mode
    default) must ride through short internal dips and only settle at the
    genuine final rest -- this is the "tune the existing state machine's
    parameters, don't build a new one" step 5 decision, verified here since
    no real camera is available to tune against directly (see
    project_workflow.md's Phase 7 section)."""
    # three "signs" (frames 5-14, 20-29, 35-44), each separated by a 5-frame
    # dip (well under settle_frames=15), then a real 20-frame final rest --
    # the last motion frame is index 44
    energies = ([0.0] * 5 + [1.0] * 10 + [0.0] * 5 + [1.0] * 10 + [0.0] * 5
                + [1.0] * 10 + [0.0] * 20)
    last_motion_frame = 44
    buf = CaptureBuffer(_energy_fn(energies), motion_threshold=0.5,
                         settle_frames=15, preroll=3, max_frames=200)
    for i in range(last_motion_frame + 1):  # through the sentence's own last sign
        buf.append(i)
        assert buf.state != "settled", (
            f"settled prematurely at frame {i}, before the sentence itself even "
            f"finished -- a brief inter-sign pause was misread as the end")
    # after the sentence ends, correctly needs settle_frames MORE quiet frames
    # before settling -- not settled yet, not stuck forever either
    for i in range(last_motion_frame + 1, last_motion_frame + 1 + 14):
        buf.append(i)
        assert buf.state != "settled"
    buf.append(last_motion_frame + 15)  # the 15th consecutive quiet frame
    assert buf.state == "settled"


def test_sentence_mode_settle_frames_too_small_settles_mid_sentence():
    """The negative case: confirms the risk step 5's plan flagged is real,
    not hypothetical -- the SAME energy pattern above, but with a
    single-sign-scale settle_frames, DOES end capture at the first inter-sign
    pause. This is why sentence mode needs its own larger default, not the
    single-sign default reused."""
    energies = ([0.0] * 5 + [1.0] * 10 + [0.0] * 5 + [1.0] * 10 + [0.0] * 5
                + [1.0] * 10 + [0.0] * 20)
    buf = CaptureBuffer(_energy_fn(energies), motion_threshold=0.5,
                         settle_frames=4, preroll=3, max_frames=200)
    for i in range(25):  # well before the sentence actually ends
        buf.append(i)
        if buf.state == "settled":
            break
    assert buf.state == "settled"  # settled too early, mid-sentence


def test_energy_fn_failure_degrades_to_plain_trailing_window():
    def raising_fn(frames):
        raise RuntimeError("no hand blocks")
    buf = CaptureBuffer(raising_fn, motion_threshold=0.5, settle_frames=4, preroll=3, max_frames=10)
    for i in range(15):
        buf.append(i)
    assert buf._energy_ok is False
    assert buf.state == "active"  # degraded mode: active once >= preroll frames buffered
    assert len(buf.frames) <= 10  # still respects the max_frames cap


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  OK   {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
