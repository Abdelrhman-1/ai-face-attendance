import base64
import os
from datetime import datetime, timedelta

import binascii
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from database import (
    add_person,
    delete_person,
    get_all_persons_from_db,
    get_attendance_records,
    get_cached_persons,
    load_cache_from_db,
    log_attendance,
)
from face_engine import (
    find_match,
    get_face_encoding_from_frame,
    get_face_encoding_from_frame_strict,
)

app = Flask(__name__)
load_cache_from_db()

# Testing value; set to timedelta(hours=8) for a real shift length.
SHIFT_DURATION = timedelta(minutes=1)

# Per-person check-in/check-out state for the current day, kept in memory
# since it's derived data and doesn't need to survive a server restart.
daily_attendance_state: dict[int, dict] = {}


# Returns today's date as a string key for grouping attendance state.
def get_today_key() -> str:
    """Return today's date as a YYYY-MM-DD string."""
    return datetime.now().strftime("%Y-%m-%d")


# Decides whether a recognized person is checking in, checking out, or
# already accounted for, and logs the event when state actually changes.
def process_attendance(person_id: int, person_name: str, distance: float) -> dict:
    """Apply check-in/check-out logic for a recognized person and return the resulting event."""
    today = get_today_key()
    now = datetime.now()
    state = daily_attendance_state.get(person_id)

    if state is None or state["date"] != today:
        daily_attendance_state[person_id] = {"date": today, "check_in": now, "check_out": None}
        log_attendance(person_id, person_name, distance, status="حضور")
        return {"event": "check_in", "message": f"تم تسجيل حضور {person_name}"}

    if state["check_out"] is None:
        elapsed = now - state["check_in"]
        if elapsed >= SHIFT_DURATION:
            state["check_out"] = now
            log_attendance(person_id, person_name, distance, status="انصراف")
            return {"event": "check_out", "message": f"تم تسجيل انصراف {person_name}"}

        remaining = SHIFT_DURATION - elapsed
        total_seconds = int(remaining.total_seconds())
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return {
            "event": "already_in",
            "message": f"{person_name} مسجل حضوره (باقي {hours:02d}:{minutes:02d}:{seconds:02d})",
        }

    return {"event": "already_out", "message": f"{person_name} أنهى دوامه اليوم بالفعل"}


# Converts a base64 data URL from the browser camera into an OpenCV frame.
def decode_base64_to_frame(base64_string: str) -> np.ndarray | None:

    if not base64_string:
        return None

    try:
        if "," in base64_string:
            _, base64_string = base64_string.split(",", 1)

        img_bytes = base64.b64decode(base64_string)
        if not img_bytes:
            return None

        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame if frame is not None else None
    except (ValueError, binascii.Error):
        return None


# --- Pages ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/enroll")
def enroll_page():
    return render_template("enroll.html")


@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/manage")
def manage_page():
    return render_template("manage.html")


# --- API: enrollment ---

@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    phone = (data.get("phone") or "").strip() or None
    image_b64 = data.get("image")

    if not name:
        return jsonify({"success": False, "message": "الرجاء إدخال الاسم"}), 400
    if not image_b64:
        return jsonify({"success": False, "message": "لم يتم استلام صورة"}), 400

    frame = decode_base64_to_frame(image_b64)
    if frame is None:
        return jsonify({"success": False, "message": "⚠️ لم يتم استلام صورة صالحة، حاول مرة أخرى"}), 400

    try:
        encoding, _ = get_face_encoding_from_frame_strict(frame)
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400


    os.makedirs("known_faces", exist_ok=True)
    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_")).strip()
    filename = f"known_faces/{safe_name}_{int(datetime.now().timestamp())}.jpg"
    cv2.imwrite(filename, frame)

    new_person = add_person(name, encoding, filename, phone)

    return jsonify({
        "success": True,
        "message": f"تم تسجيل {name} بنجاح",
        "person": {"id": new_person["id"], "name": new_person["name"]},
    })


# --- API: recognition ---

@app.route("/api/recognize_live", methods=["POST"])
def api_recognize_live():
    data = request.get_json()
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"success": False, "message": "لا توجد صورة"})

    frame = decode_base64_to_frame(image_b64)
    if frame is None:
        # Invalid or empty frame; treat as "no face found" rather than an error.
        return jsonify({"success": True, "face_found": False})

    encoding, face_location = get_face_encoding_from_frame(frame)


    if encoding is None:
        return jsonify({"success": True, "face_found": False})

    known_persons = get_cached_persons()
    match, distance = find_match(encoding, known_persons)

    top, right, bottom, left = face_location
    box = {"top": top, "right": right, "bottom": bottom, "left": left}
    frame_size = {"width": frame.shape[1], "height": frame.shape[0]}

    if match is None:
        return jsonify({
            "success": True,
            "face_found": True,
            "known": False,
            "box": box,
            "frame_size": frame_size,
            "label": "غير معروف",
            "message": "شخص غير مسجل",
        })

    result = process_attendance(match["id"], match["name"], distance)

    return jsonify({
        "success": True,
        "face_found": True,
        "known": True,
        "box": box,
        "frame_size": frame_size,
        "label": match["name"],
        "event": result["event"],
        "message": result["message"],
        "distance": round(distance, 3),
    })


# --- API: reporting & management ---

@app.route("/api/records")
def api_records():
    return jsonify(get_attendance_records(limit=200))


@app.route("/api/persons")
def api_persons():
    return jsonify(get_all_persons_from_db())


@app.route("/api/persons/<int:person_id>", methods=["DELETE"])
def api_delete_person(person_id: int):
    try:
        delete_person(person_id)
        return jsonify({"success": True, "message": "تم الحذف بنجاح"})
    except Exception as e:
        return jsonify({"success": False, "message": f"فشل الحذف: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)