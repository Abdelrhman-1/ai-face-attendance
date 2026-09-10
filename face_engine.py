import cv2
import face_recognition
import numpy as np

# Empirically, matching faces score well under 0.5 and distinct faces
# score above 0.7, so 0.6 gives a safe margin between the two.
DEFAULT_MATCH_THRESHOLD = 0.6

# Detection runs on a downscaled frame for speed. 0.35 was chosen after
# the reported real-world need: several people can appear in frame at
# once, and detection time roughly scales with the square of this factor,
# so a more aggressive downscale here matters more than encoding time
# (which is dominated by the number of faces, not frame resolution).
DEFAULT_RESIZE_FACTOR = 0.35


# --- Enrollment ---

# Extracts a face encoding from a static image file (used during enrollment).
def get_face_encoding_from_path(image_path: str) -> np.ndarray:
    """Extract a face encoding from an image file on disk."""
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        raise ValueError("No face detected in the provided image.")
    if len(face_locations) > 1:
        raise ValueError("Multiple faces detected. Provide an image with exactly one face.")

    encodings = face_recognition.face_encodings(image, face_locations)
    return encodings[0]


# Extracts a face encoding from a frame, rejecting zero or multiple faces (used during enrollment).
def get_face_encoding_from_frame_strict(frame: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Extract a face encoding from a frame, rejecting ambiguous input during enrollment."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if len(face_locations) == 0:
        raise ValueError("No face detected. Make sure your face is clearly visible.")
    if len(face_locations) > 1:
        raise ValueError("Multiple faces detected. Only one person should be in frame.")

    encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    return encodings[0], face_locations[0]


# --- Recognition ---

# Extracts a face encoding from a live camera frame (used during recognition).
def get_face_encoding_from_frame(frame: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
    """Extract a face encoding from a live camera frame, if one is present."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if len(face_locations) == 0:
        return None, None

    encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    return encodings[0], face_locations[0]


# Detects and encodes every face in a frame, for simultaneous multi-person recognition.
def get_all_face_encodings_from_frame(
    frame: np.ndarray,
    resize_factor: float = DEFAULT_RESIZE_FACTOR,
) -> list[tuple[np.ndarray, tuple]]:
    """Detect and encode every face in a live camera frame.

    Detection runs on a downscaled copy of the frame for speed; returned
    face locations are scaled back up to the original frame's coordinates,
    so callers never need to know a resize happened.
    """
    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    scale = 1 / resize_factor
    scaled_locations = [
        (int(top * scale), int(right * scale), int(bottom * scale), int(left * scale))
        for top, right, bottom, left in face_locations
    ]

    return list(zip(encodings, scaled_locations))


# Compares an unknown encoding against known persons and returns the closest match within threshold.
def find_match(
    unknown_encoding: np.ndarray,
    known_persons: list[dict],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> tuple[dict | None, float | None]:
    """Find the closest known person for a given encoding, if within threshold."""
    if not known_persons:
        return None, None

    known_encodings = [person["encoding"] for person in known_persons]
    distances = face_recognition.face_distance(known_encodings, unknown_encoding)

    best_match_index = np.argmin(distances)
    best_distance = distances[best_match_index]

    if best_distance < threshold:
        return known_persons[best_match_index], float(best_distance)

    return None, float(best_distance)


# --- Debugging utilities ---

# Draws a labeled bounding box on a frame for local visual debugging.
def draw_face_box(
    frame: np.ndarray,
    face_location: tuple,
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw a labeled bounding box around a detected face for local debugging."""
    top, right, bottom, left = face_location
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
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