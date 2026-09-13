"""Live CCTV stream bridge: RTSP in, annotated MJPEG out.

Two background threads work together:

1. _camera_reader: continuously reads RTSP frames and keeps ONLY the
   latest one, discarding everything older. This is what actually fixed
   the multi-second display lag -- without it, OpenCV/FFmpeg's internal
   buffer queues up frames faster than recognition can consume them, and
   the stream drifts further behind in real time the longer it runs.

2. _camera_loop: takes whatever the latest frame is, runs a
   detect-track-recognize pipeline on it, draws the results, and
   publishes the annotated JPEG for the /video_feed endpoint.

Detection/recognition itself (a separate cost from streaming lag) is
timed and logged, since "the video shows up instantly but the box takes
a while to appear" is a different problem with different fixes -- mainly
frame resolution and dlib's upsample setting, not buffering.
"""

import os
import threading
import time
import uuid

import cv2

import attendance_logic
import camera_manager
import database
from face_engine import detect_face_locations, encode_face_at_location, find_match

# How often the expensive detector+encoder runs. Kept low now that the
# reader thread guarantees we're always working with a fresh frame --
# there's no backlog to "catch up on" anymore, so recognition can run
# as often as the hardware allows.
RECOGNITION_INTERVAL_SECONDS = 0.3

# Detection runs on dlib's HOG detector without internal upsampling
# (see face_engine.get_all_face_encodings_from_frame) -- upsampling
# roughly doubles detection time and is meant for small/far faces, not
# a close-range entrance camera.
DETECTION_UPSAMPLE_TIMES = 0

# A tracked box must overlap a fresh detection by at least this fraction
# (Intersection over Union) to be considered "the same person" rather than
# a new face requiring a fresh recognition.
IOU_MATCH_THRESHOLD = 0.3

# A resolved identity is trusted for this long before being re-verified
# against a fresh detection, as a safety net against tracker drift.
IDENTITY_TRUST_SECONDS = 4.0

# Force TCP for RTSP (more reliable than UDP through routers/switches that
# reorder or drop packets) and disable FFmpeg's internal frame buffering,
# so stale frames aren't queued up before our own reader even sees them.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

STATUS_COLOR_BGR = {
    "حاضر": (159, 214, 51),
    "متأخر": (77, 184, 255),
}
UNKNOWN_COLOR_BGR = (107, 107, 255)
NEUTRAL_COLOR_BGR = (255, 200, 91)

# --- Shared state ---
_jpeg_lock = threading.Lock()
_latest_jpeg: bytes | None = None

_frame_lock = threading.Lock()
_latest_frame = None

_capture: cv2.VideoCapture | None = None
_worker_thread: threading.Thread | None = None
_reader_thread: threading.Thread | None = None
_stop_flag = False
_active_lesson_id: int | None = None
_last_error: str | None = None
_logged_resolution = False

_tracks: dict[str, dict] = {}


def _tracker_factory():
    """Return a callable that creates a KCF tracker, or None if unavailable.

    KCF (and most OpenCV trackers) ship only with opencv-contrib-python,
    not the plain opencv-python/opencv-python-headless package. Checked
    once at import time so a missing dependency degrades gracefully
    (falls back to detecting every pass) instead of crashing.
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


def _resolve_identity(face_location: tuple, encoding) -> tuple[str, tuple]:
    """Match an already-computed encoding and check the student into the active lesson."""
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
    """Advance every active tracker by one frame; drop any that lose their target."""
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
    """Detect all faces every pass (cheap), but only encode (expensive)
    faces that genuinely need identifying: new faces, or a track whose
    trust window has expired.

    This is the critical fix over the previous version: before, the
    expensive ~300-450ms encoder ran on every detected face on every
    pass regardless of tracking, because detection and encoding were
    bundled into a single call. Splitting them means a lingering,
    already-identified face costs only the cheap detection step (~15-50ms)
    on most passes, not the full encoder.
    """
    start = time.time()
    rgb_small_frame, small_locations, full_locations = detect_face_locations(
        frame, upsample_times=DETECTION_UPSAMPLE_TIMES
    )
    detect_ms = (time.time() - start) * 1000

    now = time.time()
    matched_track_ids = set()
    encoded_count = 0

    for small_location, face_location in zip(small_locations, full_locations):
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
            track["box"] = face_location
            if TRACKING_AVAILABLE:
                tracker = _create_tracker()
                tracker.init(frame, _to_tracker_box(face_location))
                track["tracker"] = tracker

            if now - track["resolved_at"] < IDENTITY_TRUST_SECONDS:
                # Still trusted: box re-synced above, but the expensive
                # encoder is skipped entirely for this face this pass.
                continue

            # Trust expired: this is the only case an already-tracked
            # face pays the encoding cost again, and at most once per
            # IDENTITY_TRUST_SECONDS, not every 0.3s.
            encoded_count += 1
            encoding = encode_face_at_location(rgb_small_frame, small_location)
            if encoding is not None:
                label, color = _resolve_identity(face_location, encoding)
                track.update({"label": label, "color": color, "resolved_at": now})
            continue

        # No matching track: a genuinely new face, must be encoded.
        encoded_count += 1
        encoding = encode_face_at_location(rgb_small_frame, small_location)
        if encoding is None:
            continue

        label, color = _resolve_identity(face_location, encoding)
        new_track = {"box": face_location, "label": label, "color": color, "resolved_at": now, "tracker": None}
        if TRACKING_AVAILABLE:
            tracker = _create_tracker()
            tracker.init(frame, _to_tracker_box(face_location))
            new_track["tracker"] = tracker
        new_id = str(uuid.uuid4())
        _tracks[new_id] = new_track
        matched_track_ids.add(new_id)

    total_ms = (time.time() - start) * 1000
    print(f"[Stream] detect={detect_ms:.0f}ms encoded={encoded_count} total={total_ms:.0f}ms faces={len(full_locations)}")


def _camera_reader(url: str) -> None:
    """Continuously read RTSP frames and keep only the newest one.

    This is the key fix for display lag: cv2.VideoCapture/FFmpeg buffers
    frames internally, and if recognition can't keep up with the camera's
    real frame rate, that buffer grows and the stream falls further and
    further behind real time. Reading as fast as possible here and
    immediately overwriting _latest_frame (never queuing) guarantees the
    processing loop always works with the most current frame available,
    regardless of how long recognition takes.
    """
    global _latest_frame, _capture, _last_error

    while not _stop_flag:
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                _last_error = "تعذر فتح بث الكاميرا، جاري إعادة المحاولة..."
                time.sleep(1)
                continue

            _capture = cap
            _last_error = None

            while not _stop_flag:
                success, frame = cap.read()
                if not success:
                    _last_error = "انقطع الاتصال بالكاميرا، جاري إعادة الاتصال..."
                    break
                with _frame_lock:
                    _latest_frame = frame

            cap.release()
            if _capture is cap:
                _capture = None

        except Exception as e:
            _last_error = f"خطأ في قراءة الكاميرا: {e}"
            time.sleep(1)

    if _capture is not None:
        try:
            _capture.release()
        except Exception:
            pass
    _capture = None


def _camera_loop() -> None:
    """Consume the latest frame, run detect-track-recognize, and publish the result."""
    global _latest_jpeg, _reader_thread, _tracks, _last_error, _logged_resolution

    camera = database.get_active_camera()
    if not camera:
        _last_error = "لم يتم إعداد كاميرا بعد. أضف إعدادات الكاميرا أولاً."
        return

    url = camera_manager.build_stream_url(camera)
    _tracks = {}
    _logged_resolution = False

    _reader_thread = threading.Thread(target=_camera_reader, args=(url,), daemon=True)
    _reader_thread.start()

    last_recognition_time = 0.0

    while not _stop_flag:
        try:
            with _frame_lock:
                frame = _latest_frame

            if frame is None:
                time.sleep(0.01)
                continue

            frame = frame.copy()

            if not _logged_resolution:
                # Printed once so it's easy to confirm whether the sub
                # stream (small, fast) or main stream (large, slow) is
                # actually being used -- the single biggest factor in
                # detection speed.
                h, w = frame.shape[:2]
                print(f"[Stream] Camera frame resolution: {w}x{h}")
                _logged_resolution = True

            _update_trackers(frame)

            now = time.time()
            if now - last_recognition_time >= RECOGNITION_INTERVAL_SECONDS:
                last_recognition_time = now
                _run_detection_pass(frame)
                _last_error = None

            for track in _tracks.values():
                _draw_box(frame, track["box"], track["label"], track["color"])

            encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if encoded:
                with _jpeg_lock:
                    _latest_jpeg = buffer.tobytes()

        except Exception as e:
            _last_error = f"خطأ غير متوقع بمعالجة الفريم: {e}"
            time.sleep(0.05)

    if _reader_thread is not None and _reader_thread.is_alive():
        _reader_thread.join(timeout=2)
    _reader_thread = None


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
    """Signal both background loops to stop and release the camera."""
    global _stop_flag
    _stop_flag = True


def get_latest_jpeg() -> bytes | None:
    """Return the most recently annotated frame as JPEG bytes, if any."""
    with _jpeg_lock:
        return _latest_jpeg


def get_status() -> dict:
    """Return whether the stream is running and the last error, if any."""
    return {
        "running": _worker_thread is not None and _worker_thread.is_alive(),
        "error": _last_error,
    }