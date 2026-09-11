"""Live CCTV stream bridge: RTSP in, annotated MJPEG out.

Phase 1:
- RTSP reader runs independently and keeps ONLY the newest frame.
- Recognition runs in its own worker thread.
- Display/JPEG generation runs independently from recognition.
- A slow recognition operation can no longer freeze the browser video.
"""

import os
import threading
import time
import uuid

import cv2

import attendance_logic
import camera_manager
import database
from face_engine import (
    find_match,
    get_face_locations_from_frame,
    get_face_encoding_from_location,
)

# ---------------------------------------------------------------------------
# Performance settings - Phase 1
# ---------------------------------------------------------------------------

# Keep this at 1.0 for the first test so we compare fairly with the old version.
# We can reduce it later after measuring recognition time.
RECOGNITION_INTERVAL_SECONDS = 1.0

# Browser display FPS. We do not need to JPEG-encode 30+ frames per second
# on the CPU for an attendance UI.
DISPLAY_FPS = 15
DISPLAY_INTERVAL_SECONDS = 1.0 / DISPLAY_FPS

# Limit the browser preview width. Recognition still uses the original frame.
# This greatly reduces JPEG encoding cost and browser bandwidth.
DISPLAY_MAX_WIDTH = 1280

# JPEG quality for browser preview.
DISPLAY_JPEG_QUALITY = 65

# Tracker matching.
IOU_MATCH_THRESHOLD = 0.3
IDENTITY_TRUST_SECONDS = 4.0


STATUS_COLOR_BGR = {
    "حاضر": (159, 214, 51),
    "متأخر": (77, 184, 255),
}

UNKNOWN_COLOR_BGR = (107, 107, 255)
NEUTRAL_COLOR_BGR = (255, 200, 91)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_frame_lock = threading.Lock()
_tracks_lock = threading.Lock()
_stats_lock = threading.Lock()

_latest_jpeg: bytes | None = None
_latest_frame = None

# Incremented every time the reader receives a new frame.
_latest_frame_id = 0

_capture: cv2.VideoCapture | None = None

_worker_thread: threading.Thread | None = None
_reader_thread: threading.Thread | None = None
_recognition_thread: threading.Thread | None = None

_stop_event = threading.Event()

_active_lesson_id: int | None = None
_last_error: str | None = None

_tracks: dict[str, dict] = {}

# Debug/performance statistics.
_last_recognition_ms: float | None = None
_last_detection_ms: float | None = None
_last_jpeg_ms: float | None = None


# ---------------------------------------------------------------------------
# Tracker helpers
# ---------------------------------------------------------------------------

def _tracker_factory():
    """Return a callable that creates a KCF tracker, or None if unavailable."""
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create

    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create

    return None


_create_tracker = _tracker_factory()
TRACKING_AVAILABLE = _create_tracker is not None


def _to_tracker_box(face_location: tuple) -> tuple:
    """(top, right, bottom, left) -> (x, y, w, h)."""
    top, right, bottom, left = face_location
    return (left, top, right - left, bottom - top)


def _to_face_location(tracker_box: tuple) -> tuple:
    """(x, y, w, h) -> (top, right, bottom, left)."""
    x, y, w, h = tracker_box
    return (int(y), int(x + w), int(y + h), int(x))


def _iou(box_a: tuple, box_b: tuple) -> float:
    """Intersection-over-union of two face boxes."""
    top_a, right_a, bottom_a, left_a = box_a
    top_b, right_b, bottom_b, left_b = box_b

    inter_left = max(left_a, left_b)
    inter_top = max(top_a, top_b)
    inter_right = min(right_a, right_b)
    inter_bottom = min(bottom_a, bottom_b)

    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0

    intersection = (
        (inter_right - inter_left)
        * (inter_bottom - inter_top)
    )

    area_a = (right_a - left_a) * (bottom_a - top_a)
    area_b = (right_b - left_b) * (bottom_b - top_b)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_box(
    frame,
    face_location: tuple,
    label: str,
    color: tuple,
) -> None:
    top, right, bottom, left = face_location

    cv2.rectangle(
        frame,
        (left, top),
        (right, bottom),
        color,
        2,
    )

    label_top = max(0, bottom - 22)

    cv2.rectangle(
        frame,
        (left, label_top),
        (right, bottom),
        color,
        cv2.FILLED,
    )

    cv2.putText(
        frame,
        label,
        (left + 5, max(15, bottom - 6)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.5,
        (255, 255, 255),
        1,
    )


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

def _recognize_new_face(
    face_location: tuple,
    encoding,
) -> tuple[str, tuple]:
    """Recognize a face and avoid repeated DB check-ins."""

    known_students = database.get_cached_students()

    known_encodings = database.get_cached_student_encodings()

    match, distance = find_match(
        encoding,
        known_students,
        known_encodings=known_encodings,
    )

    if match is None:
        return "غير مسجل", UNKNOWN_COLOR_BGR

    label = match["name"]
    color = NEUTRAL_COLOR_BGR

    # ---------------------------------------------------------------
    # Attendance
    # ---------------------------------------------------------------

    if _active_lesson_id is not None:

        student_id = match["id"]

        # Already processed by face recognition during this lesson.
        cached_result = _face_attendance_cache.get(
            student_id
        )

        if cached_result is not None:

            return cached_result

        try:

            result = attendance_logic.check_in_student(
                student_id,
                _active_lesson_id,
                method="face",
            )

            label = (
                f"{match['name']} - "
                f"{result['status']}"
            )

            color = STATUS_COLOR_BGR.get(
                result["status"],
                color,
            )

        except attendance_logic.LessonNotStartedError:

            label = (
                f"{match['name']} "
                f"(لم يبدأ الدرس)"
            )

        # Store the result so repeated recognition does not hit SQLite.
        _face_attendance_cache[student_id] = (
            label,
            color,
        )

    return label, color

def _run_detection_pass(frame) -> None:
    """Detect faces first, then encode only faces that need recognition."""

    global _last_recognition_ms
    global _last_detection_ms

    started_at = time.perf_counter()

    # ---------------------------------------------------------------
    # STEP 1: Detection only
    # ---------------------------------------------------------------

    detection_started = time.perf_counter()

    face_locations = get_face_locations_from_frame(
        frame
    )

    _last_detection_ms = (
        time.perf_counter() - detection_started
    ) * 1000.0

    now = time.time()

    matched_track_ids = set()

    # ---------------------------------------------------------------
    # STEP 2: Reconcile detections with existing tracks
    # ---------------------------------------------------------------

    with _tracks_lock:

        for face_location in face_locations:

            matched_track_id = None

            for track_id, track in _tracks.items():

                if track_id in matched_track_ids:
                    continue

                if _iou(
                    track["box"],
                    face_location,
                ) >= IOU_MATCH_THRESHOLD:

                    matched_track_id = track_id
                    break

            # =======================================================
            # Existing tracked face
            # =======================================================

            if matched_track_id is not None:

                matched_track_ids.add(
                    matched_track_id
                )

                track = _tracks[
                    matched_track_id
                ]

                # Still trusted:
                # update location only.
                # NO FACE ENCODING.
                if (
                    now - track["resolved_at"]
                    < IDENTITY_TRUST_SECONDS
                ):

                    track["box"] = face_location

                    if TRACKING_AVAILABLE:

                        tracker = _create_tracker()

                        tracker.init(
                            frame,
                            _to_tracker_box(
                                face_location
                            ),
                        )

                        track["tracker"] = tracker

                    continue

                # ===================================================
                # Trust expired
                # ===================================================

                encoding = get_face_encoding_from_location(
                    frame,
                    face_location,
                )

                if encoding is None:
                    continue

                label, color = _recognize_new_face(
                    face_location,
                    encoding,
                )

                track.update(
                    {
                        "box": face_location,
                        "label": label,
                        "color": color,
                        "resolved_at": now,
                    }
                )

                if TRACKING_AVAILABLE:

                    tracker = _create_tracker()

                    tracker.init(
                        frame,
                        _to_tracker_box(
                            face_location
                        ),
                    )

                    track["tracker"] = tracker

                continue

            # =======================================================
            # Brand-new face
            # =======================================================

            encoding = get_face_encoding_from_location(
                frame,
                face_location,
            )

            if encoding is None:
                continue

            label, color = _recognize_new_face(
                face_location,
                encoding,
            )

            new_track = {
                "box": face_location,
                "label": label,
                "color": color,
                "resolved_at": now,
                "tracker": None,
            }

            if TRACKING_AVAILABLE:

                tracker = _create_tracker()

                tracker.init(
                    frame,
                    _to_tracker_box(
                        face_location
                    ),
                )

                new_track["tracker"] = tracker

            new_id = str(uuid.uuid4())

            _tracks[new_id] = new_track

            matched_track_ids.add(
                new_id
            )

    _last_recognition_ms = (
        time.perf_counter() - started_at
    ) * 1000.0

def _recognition_worker() -> None:
    """Recognition runs independently from browser display."""

    global _last_error

    last_processed_frame_id = -1

    while not _stop_event.is_set():

        try:

            # Get newest frame only.
            with _frame_lock:
                frame = _latest_frame
                frame_id = _latest_frame_id

            if frame is None:
                time.sleep(0.01)
                continue

            # Do not process the same frame twice.
            if frame_id == last_processed_frame_id:
                time.sleep(0.005)
                continue

            last_processed_frame_id = frame_id

            # Copy so reader can continue writing safely.
            frame_copy = frame.copy()

            _run_detection_pass(frame_copy)

            _last_error = None

            # Recognition frequency control.
            if _stop_event.wait(
                RECOGNITION_INTERVAL_SECONDS
            ):
                break

        except Exception as e:

            _last_error = (
                f"خطأ في التعرف على الوجه: {e}"
            )

            time.sleep(0.1)


# ---------------------------------------------------------------------------
# RTSP Reader
# ---------------------------------------------------------------------------

def _camera_reader(url: str) -> None:
    """Read RTSP continuously and keep ONLY the newest frame."""

    global _latest_frame
    global _latest_frame_id
    global _capture
    global _last_error

    while not _stop_event.is_set():

        try:

            os.environ[
                "OPENCV_FFMPEG_CAPTURE_OPTIONS"
            ] = (
                "rtsp_transport;tcp|"
                "fflags;nobuffer|"
                "flags;low_delay"
            )

            cap = cv2.VideoCapture(
                url,
                cv2.CAP_FFMPEG,
            )

            cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1,
            )

            if not cap.isOpened():

                _last_error = (
                    "تعذر فتح بث الكاميرا، "
                    "جاري إعادة المحاولة..."
                )

                time.sleep(1)

                continue

            _capture = cap
            _last_error = None

            while not _stop_event.is_set():

                success, frame = cap.read()

                if not success:

                    _last_error = (
                        "انقطع الاتصال بالكاميرا، "
                        "جاري إعادة الاتصال..."
                    )

                    break

                # Keep ONLY newest frame.
                with _frame_lock:

                    _latest_frame = frame
                    _latest_frame_id += 1

            cap.release()

            if _capture is cap:
                _capture = None

        except Exception as e:

            _last_error = (
                f"خطأ في قراءة الكاميرا: {e}"
            )

            time.sleep(1)

    if _capture is not None:

        try:
            _capture.release()

        except Exception:
            pass

    _capture = None


# ---------------------------------------------------------------------------
# Display / MJPEG worker
# ---------------------------------------------------------------------------

def _resize_for_display(frame):
    """Resize only the browser preview, never the recognition source."""

    height, width = frame.shape[:2]

    if width <= DISPLAY_MAX_WIDTH:
        return frame

    scale = DISPLAY_MAX_WIDTH / width

    new_width = DISPLAY_MAX_WIDTH
    new_height = int(height * scale)

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def _display_worker() -> None:
    """Create browser JPEG frames independently from recognition."""

    global _latest_jpeg
    global _last_error
    global _last_jpeg_ms

    next_frame_time = time.perf_counter()

    while not _stop_event.is_set():

        try:

            now = time.perf_counter()

            if now < next_frame_time:

                time.sleep(
                    max(
                        0.001,
                        next_frame_time - now,
                    )
                )

            next_frame_time = (
                time.perf_counter()
                + DISPLAY_INTERVAL_SECONDS
            )

            with _frame_lock:

                frame = _latest_frame

            if frame is None:

                time.sleep(0.01)
                continue

            # Copy only for display/drawing.
            frame = frame.copy()

            # Tracker updates happen in display thread so the boxes move
            # smoothly even while recognition is busy.
            if TRACKING_AVAILABLE:

                lost_ids = []

                with _tracks_lock:

                    for track_id, track in _tracks.items():

                        tracker = track.get("tracker")

                        if tracker is None:
                            continue

                        success, box = tracker.update(
                            frame
                        )

                        if not success:

                            lost_ids.append(
                                track_id
                            )

                            continue

                        track["box"] = _to_face_location(
                            box
                        )

                    for track_id in lost_ids:

                        _tracks.pop(
                            track_id,
                            None,
                        )

            # Draw current tracks.
            with _tracks_lock:

                for track in _tracks.values():

                    _draw_box(
                        frame,
                        track["box"],
                        track["label"],
                        track["color"],
                    )

            # Reduce only browser-preview size.
            frame = _resize_for_display(frame)

            jpeg_started = time.perf_counter()

            encoded, buffer = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    DISPLAY_JPEG_QUALITY,
                ],
            )

            _last_jpeg_ms = (
                time.perf_counter()
                - jpeg_started
            ) * 1000.0

            if encoded:

                with _state_lock:

                    _latest_jpeg = (
                        buffer.tobytes()
                    )

        except Exception as e:

            _last_error = (
                f"خطأ في معالجة بث الكاميرا: {e}"
            )

            time.sleep(0.05)


# ---------------------------------------------------------------------------
# Public camera API
# ---------------------------------------------------------------------------

def start_stream(
    lesson_id: int | None = None,
) -> None:
    """Start RTSP reader, display worker and recognition worker."""

    global _active_lesson_id
    global _worker_thread
    global _reader_thread
    global _recognition_thread

    _active_lesson_id = lesson_id

    # Already running.
    if (
        _worker_thread is not None
        and _worker_thread.is_alive()
    ):
        return

    _stop_event.clear()

    camera = database.get_active_camera()

    if not camera:

        global _last_error

        _last_error = (
            "لم يتم إعداد كاميرا بعد. "
            "أضف إعدادات الكاميرا أولاً."
        )

        return

    url = camera_manager.build_stream_url(
        camera
    )

    # Reset state.
    with _frame_lock:

        global _latest_frame
        global _latest_frame_id

        _latest_frame = None
        _latest_frame_id = 0

    with _tracks_lock:

        _tracks.clear()

    # -------------------------------------------------------
    # RTSP reader
    # -------------------------------------------------------

    _reader_thread = threading.Thread(
        target=_camera_reader,
        args=(url,),
        daemon=True,
        name="camera-reader",
    )

    _reader_thread.start()

    # -------------------------------------------------------
    # Recognition worker
    # -------------------------------------------------------

    _recognition_thread = threading.Thread(
        target=_recognition_worker,
        daemon=True,
        name="face-recognition",
    )

    _recognition_thread.start()

    # -------------------------------------------------------
    # Display worker
    # -------------------------------------------------------

    _worker_thread = threading.Thread(
        target=_display_worker,
        daemon=True,
        name="camera-display",
    )

    _worker_thread.start()


def stop_stream() -> None:
    """Stop all camera workers."""

    _stop_event.set()

    global _capture

    if _capture is not None:

        try:
            _capture.release()

        except Exception:
            pass

        _capture = None


def get_latest_jpeg() -> bytes | None:
    """Return the latest browser JPEG."""

    with _state_lock:

        return _latest_jpeg


def get_status() -> dict:
    """Return stream status and useful performance metrics."""

    with _stats_lock:

        recognition_ms = _last_recognition_ms
        detection_ms = _last_detection_ms
        jpeg_ms = _last_jpeg_ms

    return {
        "running": (
            _worker_thread is not None
            and _worker_thread.is_alive()
        ),
        "error": _last_error,
        "tracking_available": TRACKING_AVAILABLE,
        "recognition_ms": recognition_ms,
        "detection_ms": detection_ms,
        "jpeg_ms": jpeg_ms,
        "display_fps": DISPLAY_FPS,
    }