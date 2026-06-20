from collections import deque

import cv2
import numpy as np

EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


class LivenessChecker:
    """Detects live faces vs printed photos or phone screens."""

    def __init__(self, window: int = 30, require_blink: bool = False):
        self.window = window
        self.require_blink = require_blink
        self.positions: deque[tuple[float, float]] = deque(maxlen=window)
        self.eye_counts: deque[int] = deque(maxlen=window)
        self.laplacian_vars: deque[float] = deque(maxlen=window)
        self.frame_diffs: deque[float] = deque(maxlen=window)
        self.blink_detected = False
        self.frames_tracked = 0
        self.static_frames = 0
        self._last_center: tuple[float, float] | None = None
        self._last_gray_face: np.ndarray | None = None
        self.spoof_streak = 0

    def reset(self):
        self.positions.clear()
        self.eye_counts.clear()
        self.laplacian_vars.clear()
        self.frame_diffs.clear()
        self.blink_detected = False
        self.frames_tracked = 0
        self.static_frames = 0
        self._last_center = None
        self._last_gray_face = None
        self.spoof_streak = 0

    def update(
        self,
        gray_frame: np.ndarray,
        color_face: np.ndarray,
        raw_gray_face: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[bool, str, bool]:
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        self.positions.append((cx, cy))
        self.frames_tracked += 1

        if self._last_center is not None:
            dist = np.hypot(cx - self._last_center[0], cy - self._last_center[1])
            self.static_frames = self.static_frames + 1 if dist < 2.5 else 0
        self._last_center = (cx, cy)

        if self._last_gray_face is not None and raw_gray_face.shape == self._last_gray_face.shape:
            diff = float(np.mean(cv2.absdiff(raw_gray_face, self._last_gray_face)))
            self.frame_diffs.append(diff)
        self._last_gray_face = raw_gray_face.copy()

        face_roi = gray_frame[y : y + h, x : x + w]
        eyes = EYE_CASCADE.detectMultiScale(
            face_roi, scaleFactor=1.1, minNeighbors=3, minSize=(18, 18)
        )
        eye_count = len(eyes)
        self.eye_counts.append(eye_count)
        self._detect_blink()

        lap_var = float(cv2.Laplacian(raw_gray_face, cv2.CV_64F).var())
        self.laplacian_vars.append(lap_var)

        spoof, reason = self._check_spoof(raw_gray_face, color_face, lap_var)
        if spoof:
            self.spoof_streak += 1
            if self.spoof_streak >= 3:
                return False, reason, True
        else:
            self.spoof_streak = max(0, self.spoof_streak - 1)

        if self.frames_tracked < 8:
            return False, "Look at camera...", False

        if self.require_blink:
            if self.blink_detected:
                return True, "Blink verified", False
            if eye_count >= 1:
                return False, "Blink once to mark attendance", False
            return False, "Open eyes, then blink", False

        if self.blink_detected:
            return True, "Live face verified", False

        motion = self._motion_variance()
        avg_lap = float(np.mean(self.laplacian_vars)) if self.laplacian_vars else 0
        if motion >= 1.2 and 18 <= avg_lap <= 950:
            return True, "Live face verified", False

        if self._is_static_photo():
            return False, "Phone/photo detected", True

        return False, "Blink or move head slightly", False

    def _detect_blink(self):
        if len(self.eye_counts) < 5:
            return

        counts = list(self.eye_counts)
        for i in range(len(counts) - 2):
            if counts[i] >= 1 and counts[i + 1] == 0 and counts[i + 2] >= 1:
                self.blink_detected = True
                return

        if len(counts) >= 8:
            open_frames = sum(1 for c in counts[-8:] if c >= 1)
            closed_frames = sum(1 for c in counts[-8:] if c == 0)
            if open_frames >= 3 and closed_frames >= 1 and counts[-1] >= 1:
                self.blink_detected = True

    def _is_static_photo(self) -> bool:
        if len(self.frame_diffs) < 20:
            return False
        avg_diff = float(np.mean(self.frame_diffs))
        motion = self._motion_variance()
        return avg_diff < 1.2 and motion < 1.5 and self.frames_tracked > 25

    def _motion_variance(self) -> float:
        if len(self.positions) < 5:
            return 0.0
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        return float(np.var(xs) + np.var(ys))

    def _check_spoof(
        self, gray_face: np.ndarray, color_face: np.ndarray, lap_var: float
    ) -> tuple[bool, str]:
        if lap_var < 15:
            return True, "Flat print detected"

        if self._is_static_photo() and self.frames_tracked > 25:
            return True, "Phone/photo detected"

        h, w = gray_face.shape
        if h > 0 and w > 0:
            small = cv2.resize(gray_face, (128, 128))
            fft = np.abs(np.fft.fftshift(np.fft.fft2(small)))
            cy, cx = 64, 64
            core = fft[cy - 16 : cy + 16, cx - 16 : cx + 16].mean()
            outer = fft.mean()
            if core > 0 and outer / core > 2.8 and lap_var > 450:
                return True, "Phone screen detected"

        if color_face is not None and color_face.size > 0:
            b, g, r = cv2.split(color_face)
            b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
            if b_mean > r_mean * 1.15 and g_mean > r_mean * 1.08 and lap_var > 320:
                return True, "Screen glow detected"

        return False, ""
