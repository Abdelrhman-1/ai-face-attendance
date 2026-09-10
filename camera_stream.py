"""Live CCTV stream bridge: RTSP in, annotated MJPEG out.

Uses a detect-track-recognize pipeline rather than running the expensive
face detector and encoder on every frame:

- Every frame: cheap trackers (KCF) follow already-identified faces, so
  boxes move smoothly with the person instead of freezing between
  recognition passes.
- Every RECOGNITION_INTERVAL_SECONDS: the expensive detector+encoder runs
  once, to (a) catch newly-appeared faces and (b) drop trackers that have
  drifted off their target. A face that's already being tracked with a
  resolved identity is not re-encoded on every pass -- it's matched to its
  existing track by bounding-box overlap and left alone, so a person who
  lingers in frame is only recognized once, not every interval.
"""

import threading
import time
import uuid

import cv2

import attendance_logic
import camera_manager
import database
from face_engine import find_match, get_all_face_encodings_from_frame

# How often the expensive detector+encoder runs. Between passes, cheap
# trackers keep boxes visually attached to moving people.
RECOGNITION_INTERVAL_SECONDS = 0.4

# A tracked box must overlap a fresh detection by at least this fraction
# (Intersection over Union) to be considered "the same person" rather than
# a new face requiring a fresh recognition.
IOU_MATCH_THRESHOLD = 0.3

# A resolved identity is trusted for this long before being re-verified
# against a fresh detection, as a safety net against tracker drift.
IDENTITY_TRUST_SECONDS = 4.0

STATUS_COLOR_BGR = {
    "حاضر": (159, 214, 51),
    "متأخر": (77, 184, 255),
}
UNKNOWN_COLOR_BGR = (107, 107, 255)
NEUTRAL_COLOR_BGR = (255, 200, 91)

_lock = threading.Lock()
_latest_jpeg: bytes | None = None
_capture: cv2.VideoCapture | None = None
_worker_thread: threading.Thread | None = None
_stop_flag = False
_active_lesson_id: int | None = None
_last_error: str | None = None

# Active tracks: track_id -> {tracker, box, label, color, resolved_at}
_tracks: dict[str, dict] = {}


def _tracker_factory():
    """Return a callable that creates a KCF tracker, or None if unavailable.

    KCF (and most OpenCV trackers) ship only with opencv-contrib-python,
    not the plain opencv-python/opencv-python-headless package listed in
    requirements.txt. Checked once at import time so a missing dependency
    degrades gracefully (falls back to detecting every pass, same as
    before tracking was added) instead of crashing the background thread
    the first time a track is created.
    """
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create
    return None


_create_tracker = _tracker_factory()
TRACKING_AVAILABLE = _create_tracker is not None


def _to_tracker_box(face_location: tuple) -> tuple:
    """Convert (top, right, bottom, left) to OpenCV's (x, y, w, h) box format."""
    top, right, bottom, left = face_location
    return (left, top, right - left, bottom - top)


def _to_face_location(tracker_box: tuple) -> tuple:
    """Convert OpenCV's (x, y, w, h) box format back to (top, right, bottom, left)."""
    x, y, w, h = tracker_box
    return (int(y), int(x + w), int(y + h), int(x))


def _iou(box_a: tuple, box_b: tuple) -> float:
    """Intersection-over-union of two (top, right, bottom, left) boxes."""
    top_a, right_a, bottom_a, left_a = box_a
    top_b, right_b, bottom_b, left_b = box_b

    inter_left, inter_top = max(left_a, left_b), max(top_a, top_b)
    inter_right, inter_bottom = min(right_a, right_b), min(bottom_a, bottom_b)

    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0

    intersection = (inter_right - inter_left) * (inter_bottom - inter_top)
    area_a = (right_a - left_a) * (bottom_a - top_a)
    area_b = (right_b - left_b) * (bottom_b - top_b)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _draw_box(frame, face_location: tuple, label: str, color: tuple) -> None:
    top, right, bottom, left = face_location
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.rectangle(frame, (left, bottom - 22), (right, bottom), color, cv2.FILLED)
    cv2.putText(
        frame, label, (left + 5, bottom - 6),
        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1,
    )


def _recognize_new_face(face_location: tuple, encoding) -> tuple[str, tuple]:
    """Run the full recognition + check-in flow for a face with no existing track."""
    known_students = database.get_cached_students()
    match, _distance = find_match(encoding, known_students)

    if match is None:
        return "غير مسجل", UNKNOWN_COLOR_BGR

    label, color = match["name"], NEUTRAL_COLOR_BGR
    if _active_lesson_id is not None:
        try:
            result = attendance_logic.check_in_student(match["id"], _active_lesson_id, method="face")
            label = f"{match['name']} - {result['status']}"
            color = STATUS_COLOR_BGR.get(result["status"], color)
        except attendance_logic.LessonNotStartedError:
            label = f"{match['name']} (لم يبدأ الدرس)"

    return label, color


def _update_trackers(frame) -> None:
    """Advance every active tracker by one frame; drop any that lose their target.

    A no-op if tracking is unavailable (see TRACKING_AVAILABLE) -- boxes
    then simply hold their last detected position until the next
    detection pass, rather than the app crashing.
    """
    if not TRACKING_AVAILABLE:
        return

    lost_ids = []
    for track_id, track in _tracks.items():
        success, box = track["tracker"].update(frame)
        if not success:
            lost_ids.append(track_id)
            continue
        track["box"] = _to_face_location(box)

    for track_id in lost_ids:
        del _tracks[track_id]


def _run_detection_pass(frame) -> None:
    """Detect all faces, reconcile with existing tracks, and recognize new ones.

    A detected face overlapping an existing, recently-resolved track is left
    alone (no re-encoding). A track whose trust window has expired is
    re-recognized but updated in place -- never replaced with a second,
    duplicate track for the same physical face. Only a face with no
    matching track at all gets a brand new one.
    """
    detections = get_all_face_encodings_from_frame(frame)
    now = time.time()
    matched_track_ids = set()

    for encoding, face_location in detections:
        matched_track_id = None
        for track_id, track in _tracks.items():
            if track_id in matched_track_ids:
                continue
            if _iou(track["box"], face_location) >= IOU_MATCH_THRESHOLD:
                matched_track_id = track_id
                break

        if matched_track_id is not None:
            matched_track_ids.add(matched_track_id)
            track = _tracks[matched_track_id]

            if now - track["resolved_at"] < IDENTITY_TRUST_SECONDS:
                # Still trusted: just re-sync position, skip re-recognition.
                track["box"] = face_location
                if TRACKING_AVAILABLE:
                    tracker = _create_tracker()
                    tracker.init(frame, _to_tracker_box(face_location))
                    track["tracker"] = tracker
                continue

            # Trust expired: re-recognize, but update this same track --
            # never create a second entry for a face that's already tracked.
            label, color = _recognize_new_face(face_location, encoding)
            track.update({"box": face_location, "label": label, "color": color, "resolved_at": now})
            if TRACKING_AVAILABLE:
                tracker = _create_tracker()
                tracker.init(frame, _to_tracker_box(face_location))
                track["tracker"] = tracker
            continue

        # No matching track at all: genuinely new face.
        label, color = _recognize_new_face(face_location, encoding)
        new_track = {"box": face_location, "label": label, "color": color, "resolved_at": now, "tracker": None}
        if TRACKING_AVAILABLE:
            tracker = _create_tracker()
            tracker.init(frame, _to_tracker_box(face_location))
            new_track["tracker"] = tracker
        new_id = str(uuid.uuid4())
        _tracks[new_id] = new_track
        matched_track_ids.add(new_id)


def _camera_loop() -> None:
    global _latest_jpeg, _capture, _last_error, _tracks

    camera = database.get_active_camera()
    if not camera:
        _last_error = "لم يتم إعداد كاميرا بعد. أضف إعدادات الكاميرا أولاً."
        return

    url = camera_manager.build_stream_url(camera)
    _capture = cv2.VideoCapture(url)
    if not _capture.isOpened():
        _last_error = "تعذر فتح بث الكاميرا. تحقق من العنوان، المنفذ، ومسار البث بالإعدادات."
        return

    _last_error = None
    _tracks = {}
    last_recognition_time = 0.0

    while not _stop_flag:
        try:
            success, frame = _capture.read()
            if not success:
                _last_error = "انقطع الاتصال بالكاميرا، جاري إعادة المحاولة..."
                time.sleep(0.5)
                _capture.release()
                _capture = cv2.VideoCapture(url)
                continue

            _update_trackers(frame)

            now = time.time()
            if now - last_recognition_time >= RECOGNITION_INTERVAL_SECONDS:
                last_recognition_time = now
                _run_detection_pass(frame)
                _last_error = None  # a successful pass clears any prior transient error

            for track in _tracks.values():
                _draw_box(frame, track["box"], track["label"], track["color"])

            encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if encoded:
                with _lock:
                    _latest_jpeg = buffer.tobytes()
        except Exception as e:
            # Anything unexpected here (e.g. a tracker/recognition edge case)
            # would otherwise kill this background thread silently, leaving
            # the stream frozen with no visible explanation. Surfacing it
            # through _last_error lets the frontend show a real message
            # instead of just "not running".
            _last_error = f"خطأ غير متوقع بمعالجة الفريم: {e}"
            time.sleep(0.2)

    _capture.release()
    _capture = None


def start_stream(lesson_id: int | None = None) -> None:
    """Start the shared camera stream if not already running, and set the
    lesson that recognized faces should be checked into.
    """
    global _active_lesson_id, _worker_thread, _stop_flag

    _active_lesson_id = lesson_id

    if _worker_thread is not None and _worker_thread.is_alive():
        return

    _stop_flag = False
    _worker_thread = threading.Thread(target=_camera_loop, daemon=True)
    _worker_thread.start()


def stop_stream() -> None:
    """Signal the background loop to stop and release the camera."""
    global _stop_flag
    _stop_flag = True


def get_latest_jpeg() -> bytes | None:
    """Return the most recently annotated frame as JPEG bytes, if any."""
    with _lock:
        return _latest_jpeg


def get_status() -> dict:
    """Return whether the stream is running and the last error, if any."""
    return {
        "running": _worker_thread is not None and _worker_thread.is_alive(),
        "error": _last_error,
    }