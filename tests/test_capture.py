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
