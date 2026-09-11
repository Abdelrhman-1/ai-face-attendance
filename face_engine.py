import cv2
import face_recognition
import numpy as np


DEFAULT_MATCH_THRESHOLD = 0.6

# Detection runs on a smaller image for CPU performance.
DEFAULT_RESIZE_FACTOR = 0.35


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

def get_face_encoding_from_path(image_path: str) -> np.ndarray:
    """Extract one face encoding from an enrollment image."""
    image = face_recognition.load_image_file(image_path)

    face_locations = face_recognition.face_locations(
        image,
        number_of_times_to_upsample=0,
        model="hog",
    )

    if len(face_locations) == 0:
        raise ValueError("No face detected in the provided image.")

    if len(face_locations) > 1:
        raise ValueError(
            "Multiple faces detected. "
            "Provide an image with exactly one face."
        )

    encodings = face_recognition.face_encodings(
        image,
        face_locations,
    )

    return encodings[0]


def get_face_encoding_from_frame_strict(
    frame: np.ndarray,
) -> tuple[np.ndarray, tuple]:
    """Extract exactly one face encoding from a frame."""
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    face_locations = face_recognition.face_locations(
        rgb_frame,
        number_of_times_to_upsample=0,
        model="hog",
    )

    if len(face_locations) == 0:
        raise ValueError(
            "No face detected. Make sure your face is clearly visible."
        )

    if len(face_locations) > 1:
        raise ValueError(
            "Multiple faces detected. "
            "Only one person should be in frame."
        )

    encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations,
    )

    return encodings[0], face_locations[0]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def get_face_locations_from_frame(
    frame: np.ndarray,
    resize_factor: float = DEFAULT_RESIZE_FACTOR,
) -> list[tuple]:
    """Detect faces only.

    Detection runs on a downscaled frame.
    Returned coordinates are scaled back to original-frame coordinates.

    IMPORTANT:
    This function does NOT calculate face encodings.
    """

    if frame is None or frame.size == 0:
        return []

    # Downscale for faster detection.
    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=resize_factor,
        fy=resize_factor,
        interpolation=cv2.INTER_AREA,
    )

    rgb_small_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB,
    )

    face_locations = face_recognition.face_locations(
        rgb_small_frame,
        number_of_times_to_upsample=0,
        model="hog",
    )

    scale = 1.0 / resize_factor

    scaled_locations = [
        (
            int(top * scale),
            int(right * scale),
            int(bottom * scale),
            int(left * scale),
        )
        for top, right, bottom, left in face_locations
    ]

    return scaled_locations


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def get_face_encoding_from_location(
    frame: np.ndarray,
    face_location: tuple,
) -> np.ndarray | None:
    """Encode ONE detected face.

    Only the detected face crop is sent to face_recognition.
    This is much cheaper than encoding all faces on every pass.
    """

    if frame is None or frame.size == 0:
        return None

    top, right, bottom, left = face_location

    height, width = frame.shape[:2]

    # Clamp coordinates safely.
    top = max(0, min(height, top))
    bottom = max(0, min(height, bottom))
    left = max(0, min(width, left))
    right = max(0, min(width, right))

    if right <= left or bottom <= top:
        return None

    # Add a small margin around the face.
    face_width = right - left
    face_height = bottom - top

    margin_x = int(face_width * 0.25)
    margin_y = int(face_height * 0.35)

    crop_left = max(0, left - margin_x)
    crop_right = min(width, right + margin_x)

    crop_top = max(0, top - margin_y)
    crop_bottom = min(height, bottom + margin_y)

    face_crop = frame[
        crop_top:crop_bottom,
        crop_left:crop_right,
    ]

    if face_crop.size == 0:
        return None

    rgb_crop = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB,
    )

    # Face location relative to crop.
    local_location = (
        top - crop_top,
        right - crop_left,
        bottom - crop_top,
        left - crop_left,
    )

    encodings = face_recognition.face_encodings(
        rgb_crop,
        [local_location],
        num_jitters=1,
    )

    if not encodings:
        return None

    return encodings[0]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def find_match(
    unknown_encoding: np.ndarray,
    known_persons: list[dict],
    known_encodings: np.ndarray | None = None,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> tuple[dict | None, float | None]:
    """Find closest known person using a pre-built NumPy matrix."""

    if not known_persons:
        return None, None

    if known_encodings is None:
        known_encodings = np.asarray(
            [person["encoding"] for person in known_persons],
            dtype=np.float64,
        )

    if known_encodings.size == 0:
        return None, None

    # Euclidean distance between unknown embedding and all known embeddings.
    distances = np.linalg.norm(
        known_encodings - unknown_encoding,
        axis=1,
    )

    best_match_index = int(
        np.argmin(distances)
    )

    best_distance = float(
        distances[best_match_index]
    )

    if best_distance < threshold:
        return (
            known_persons[best_match_index],
            best_distance,
        )

    return None, best_distance

# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

def draw_face_box(
    frame: np.ndarray,
    face_location: tuple,
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:

    top, right, bottom, left = face_location

    cv2.rectangle(
        frame,
        (left, top),
        (right, bottom),
        color,
        2,
    )

    cv2.rectangle(
        frame,
        (left, bottom - 25),
        (right, bottom),
        color,
        cv2.FILLED,
    )

    cv2.putText(
        frame,
        label,
        (left + 6, bottom - 6),
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    return frame

def get_all_face_encodings_from_frame(
    frame: np.ndarray,
    resize_factor: float = DEFAULT_RESIZE_FACTOR,
) -> list[tuple[np.ndarray, tuple]]:
    """Compatibility helper for the browser webcam recognition path."""

    locations = get_face_locations_from_frame(
        frame,
        resize_factor=resize_factor,
    )

    results = []

    for location in locations:

        encoding = get_face_encoding_from_location(
            frame,
            location,
        )

        if encoding is not None:
            results.append(
                (encoding, location)
            )

    return results