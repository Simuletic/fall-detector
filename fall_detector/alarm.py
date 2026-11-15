"""'Stays down' alarm: fire only when someone has looked like they are lying on the floor for N seconds in a row.

Bending down to pick something up, tying shoes or kneeling usually lasts about a second; a person who has fallen
stays down. The persistent score at time t is the LOWEST single-frame score over the last `seconds`, so one
confident frame is not enough: every frame in the window has to agree.

    alarm = StaysDownAlarm(threshold=0.5, seconds=2.0)
    for t, frame in frames:                       # t in seconds
        fired, persistent = alarm.update(t, detector.frame_score(frame))
"""
from __future__ import annotations

from collections import deque


class StaysDownAlarm:
    def __init__(self, threshold: float = 0.5, seconds: float = 2.0):
        self.threshold = threshold
        self.seconds = seconds
        self.history: deque[tuple[float, float]] = deque()
        self.active = False

    def reset(self) -> None:
        self.history.clear()
        self.active = False

    def update(self, t: float, score: float) -> tuple[bool, float | None]:
        """Add one frame score at time t (seconds). Returns (alarm_fired_now, persistent_score or None until the
        window is full)."""
        self.history.append((t, score))
        while self.history and self.history[0][0] < t - self.seconds:
            self.history.popleft()
        if t - self.history[0][0] < self.seconds * 0.9:  # not enough history yet (allows a little frame jitter)
            return False, None
        persistent = min(s for _, s in self.history)
        fired = persistent >= self.threshold and not self.active
        self.active = persistent >= self.threshold
        return fired, persistent
