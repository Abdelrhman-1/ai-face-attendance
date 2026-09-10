import queue
import threading
from urllib.parse import quote

import cv2

import database

# cv2.VideoCapture can hang for a long time against an unreachable RTSP
# address (no built-in timeout), so connection attempts run in a background
# thread and are abandoned if they exceed this limit, rather than blocking
# the request indefinitely.
DEFAULT_TIMEOUT_SECONDS = 6


def build_stream_url(camera: dict) -> str:
    """Build the full RTSP connection URL from stored camera settings.

    Username/password are URL-encoded since IP camera credentials often
    contain characters (@, :, /) that would otherwise break the URL.
    """
    username = quote(camera.get("username") or "", safe="")
    password = quote(camera.get("password") or "", safe="")
    credentials = f"{username}:{password}@" if username or password else ""
    path = camera.get("stream_path") or ""
    protocol = camera.get("protocol") or "rtsp"
    return f"{protocol}://{credentials}{camera['ip_address']}:{camera['port']}{path}"


def _attempt_connection(url: str, result_queue: "queue.Queue") -> None:
    """Try to open the stream and read one frame; runs in a worker thread."""
    capture = cv2.VideoCapture(url)
    if not capture.isOpened():
        result_queue.put((False, "تعذر فتح الاتصال بالكاميرا"))
        return

    success, _ = capture.read()
    capture.release()

    if success:
        result_queue.put((True, "تم الاتصال بنجاح واستقبال الصورة"))
    else:
        result_queue.put((False, "تم فتح الاتصال لكن تعذر استقبال صورة من الكاميرا"))


def test_connection(camera: dict, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Attempt to connect to the camera and read one frame, with a hard timeout.

    The worker thread is left running (as a daemon) if it doesn't finish in
    time rather than being force-killed, since Python threads can't be
    safely terminated; it will simply exit on its own once the underlying
    connection attempt eventually fails or succeeds.
    """
    url = build_stream_url(camera)
    result_queue: "queue.Queue" = queue.Queue()

    worker = threading.Thread(target=_attempt_connection, args=(url, result_queue), daemon=True)
    worker.start()
    worker.join(timeout=timeout_seconds)

    if worker.is_alive():
        status, message = "disconnected", "انتهت مهلة الاتصال بالكاميرا (تحقق من العنوان والمنفذ)"
    else:
        success, message = result_queue.get()
        status = "connected" if success else "disconnected"

    if camera.get("id") is not None:
        database.update_camera_status(camera["id"], status)

    return {"status": status, "message": message}


def get_active_stream_url() -> str | None:
    """Return the currently configured active camera's stream URL, if any."""
    camera = database.get_active_camera()
    return build_stream_url(camera) if camera else None