import pickle
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from liveness import LivenessChecker

DATA_DIR = Path(__file__).parent / "data"
FACES_DIR = DATA_DIR / "faces"
MODEL_PATH = DATA_DIR / "face_model.yml"
LABELS_PATH = DATA_DIR / "labels.pkl"

FACE_SIZE = (200, 200)
CONFIDENCE_THRESHOLD = 75
DISPLAY_THRESHOLD = 95
VOTE_FRAMES = 6
VOTE_MIN_MATCHES = 3

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def ensure_dirs():
    FACES_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_face(gray_face: np.ndarray) -> np.ndarray:
    normalized = CLAHE.apply(gray_face)
    return cv2.resize(normalized, FACE_SIZE)


def detect_faces(gray_frame, scale_factor=1.1, min_neighbors=5):
    faces = FACE_CASCADE.detectMultiScale(
        gray_frame,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return []
    return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)


def detect_face(gray_frame):
    faces = detect_faces(gray_frame)
    return faces[0] if faces else None


def draw_face_box(
    frame,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    color: tuple[int, int, int],
):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    label_y = max(y - 10, text_h + 10)
    cv2.rectangle(
        frame,
        (x, label_y - text_h - 8),
        (x + text_w + 12, label_y + baseline),
        color,
        -1,
    )
    cv2.putText(frame, label, (x + 6, label_y - 2), font, scale, (255, 255, 255), thickness)


class RecognitionTracker:
    """Stable recognition via multi-frame voting."""

    def __init__(
        self,
        vote_size: int = VOTE_FRAMES,
        min_matches: int = VOTE_MIN_MATCHES,
        require_blink: bool = True,
    ):
        self.votes: deque[str | None] = deque(maxlen=vote_size)
        self.min_matches = min_matches
        self.liveness = LivenessChecker(require_blink=require_blink)
        self.last_match_id: str | None = None
        self.last_confidence: float = 999.0

    def reset(self):
        self.votes.clear()
        self.liveness.reset()
        self.last_match_id = None
        self.last_confidence = 999.0

    def update(
        self,
        gray_frame: np.ndarray,
        color_frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        recognizer,
        label_map: dict,
    ) -> dict:
        x, y, w, h = bbox
        gray_roi = gray_frame[y : y + h, x : x + w]
        color_roi = color_frame[y : y + h, x : x + w]
        raw_gray = cv2.resize(gray_roi, FACE_SIZE)
        face = preprocess_face(gray_roi)

        match_id, confidence, confirmed = recognize_face(face, recognizer, label_map)
        if match_id and confidence <= DISPLAY_THRESHOLD:
            self.last_match_id = match_id
            self.last_confidence = confidence

        self.votes.append(match_id if confirmed else None)

        live, live_msg, is_spoof = self.liveness.update(
            gray_frame, color_roi, raw_gray, bbox
        )

        stable_id = self._stable_match()
        display_id = stable_id or (
            self.last_match_id if self.last_confidence <= DISPLAY_THRESHOLD else None
        )

        return {
            "student_id": stable_id,
            "display_id": display_id,
            "raw_id": match_id if confirmed else None,
            "best_id": self.last_match_id,
            "confidence": self.last_confidence,
            "live": live,
            "live_msg": live_msg,
            "is_spoof": is_spoof,
            "blink_done": self.liveness.blink_detected,
            "votes": len([v for v in self.votes if v == stable_id]) if stable_id else 0,
        }

    def _stable_match(self) -> str | None:
        ids = [v for v in self.votes if v is not None]
        if len(ids) < self.min_matches:
            return None
        best_id, count = Counter(ids).most_common(1)[0]
        return best_id if count >= self.min_matches else None


def get_registered_face_ids() -> set[str]:
    ensure_dirs()
    return {
        p.name for p in FACES_DIR.iterdir() if p.is_dir() and any(p.glob("*.jpg"))
    }


def capture_face_samples(student_id: str, sample_count: int = 50) -> tuple[bool, str]:
    ensure_dirs()
    student_dir = FACES_DIR / student_id
    student_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        return False, "Could not open camera"

    captured = 0
    frame_count = 0

    while captured < sample_count:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)

        display = frame.copy()
        cv2.putText(
            display,
            f"Register: {student_id} | Samples: {captured}/{sample_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display,
            "Slowly turn head. Blink once. Press Q to cancel.",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        if len(faces) > 0:
            x, y, w, h = faces[0]
            draw_face_box(display, x, y, w, h, "Capturing...", (0, 255, 0))

            if frame_count % 2 == 0:
                face_img = preprocess_face(gray[y : y + h, x : x + w])
                filename = student_dir / f"{captured:03d}.jpg"
                cv2.imwrite(str(filename), face_img)
                captured += 1

        cv2.imshow("Register Face", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return False, "Registration cancelled"

    cap.release()
    cv2.destroyAllWindows()

    if captured < 20:
        return False, "Not enough face samples captured"

    return True, f"Captured {captured} face samples"


def load_training_data():
    ensure_dirs()
    faces = []
    labels = []
    label_map = {}
    next_label = 0

    for student_dir in sorted(FACES_DIR.iterdir()):
        if not student_dir.is_dir():
            continue
        images = list(student_dir.glob("*.jpg"))
        if not images:
            continue

        student_id = student_dir.name
        label_map[next_label] = student_id

        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != FACE_SIZE:
                img = cv2.resize(img, FACE_SIZE)
            faces.append(img)
            labels.append(next_label)

        next_label += 1

    return faces, labels, label_map


def train_model() -> tuple[bool, str]:
    faces, labels, label_map = load_training_data()

    if len(faces) == 0:
        return False, "No face data found. Register students first."

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(faces, np.array(labels))

    ensure_dirs()
    recognizer.write(str(MODEL_PATH))
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    return True, f"Model trained on {len(label_map)} student(s), {len(faces)} samples"


def load_model():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    with open(LABELS_PATH, "rb") as f:
        label_map = pickle.load(f)

    return recognizer, label_map


def recognize_face(
    gray_face,
    recognizer,
    label_map,
    confidence_threshold=CONFIDENCE_THRESHOLD,
):
    if recognizer is None or not label_map:
        return None, 999.0, False

    label, confidence = recognizer.predict(gray_face)
    confidence = float(confidence)
    student_id = label_map.get(label)
    confirmed = student_id is not None and confidence <= confidence_threshold
    return student_id, confidence, confirmed
