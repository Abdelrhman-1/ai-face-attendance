import base64
import binascii
import os
import time
import uuid

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

import attendance_logic
import camera_manager
import camera_stream
import database
import excel_export
from face_engine import (
    find_match,
    get_all_face_encodings_from_frame,
    get_face_encoding_from_frame_strict,
)

app = Flask(__name__)
database.load_student_cache()


# Converts a base64 data URL from the browser camera into an OpenCV frame.
def decode_base64_to_frame(base64_string: str) -> np.ndarray | None:
    """Decode a base64 image (optionally a data URL) into a BGR OpenCV frame.

    Returns None instead of raising when the payload is empty or malformed,
    since the browser occasionally posts invalid frames during camera
    warm-up or tab switches.
    """
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


def save_face_photo(frame: np.ndarray) -> str:
    """Save a captured frame to disk under a random filename and return its path.

    A UUID-based filename avoids Windows codepage issues with non-ASCII
    names, since the display name is stored separately in the database.
    """
    os.makedirs("known_faces", exist_ok=True)
    filename = f"known_faces/{uuid.uuid4().hex}.jpg"
    cv2.imwrite(filename, frame)
    return filename


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


@app.route("/week")
def week_page():
    return render_template("week.html")


@app.route("/manage")
def manage_page():
    return render_template("manage.html")


@app.route("/manage/<int:student_id>")
def student_detail_page(student_id: int):
    return render_template("student_detail.html", student_id=student_id)


@app.route("/settings")
def settings_page():
    return render_template("settings.html")


# --- API: student enrollment & management ---

@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    national_id = (data.get("national_id") or "").strip()
    phone = (data.get("phone") or "").strip() or None
    image_b64 = data.get("image")

    if not name:
        return jsonify({"success": False, "message": "الرجاء إدخال الاسم"}), 400
    if not national_id:
        return jsonify({"success": False, "message": "الرجاء إدخال رقم الهوية/الوثيقة"}), 400

    frame = decode_base64_to_frame(image_b64)
    if frame is None:
        return jsonify({"success": False, "message": "لم يتم استلام صورة صالحة، حاول مرة أخرى"}), 400

    try:
        encoding, _ = get_face_encoding_from_frame_strict(frame)
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400

    filename = save_face_photo(frame)

    try:
        new_student = database.add_student(name, national_id, encoding, filename, phone)
    except database.DuplicateNationalIdError:
        return jsonify({"success": False, "message": "رقم الهوية/الوثيقة مستخدم مسبقاً"}), 409

    return jsonify({"success": True, "message": f"تم تسجيل {name} بنجاح", "student": new_student})


@app.route("/api/students/search")
def api_search_students():
    query = request.args.get("q", "").strip()
    students = database.search_students(query) if query else database.get_all_students()
    return jsonify(students)


@app.route("/api/students/<int:student_id>", methods=["GET"])
def api_get_student(student_id: int):
    student = database.get_student_by_id(student_id)
    if not student:
        return jsonify({"success": False, "message": "الطالب غير موجود"}), 404
    history = database.get_student_history(student_id)
    return jsonify({"success": True, "student": student, "history": history})


@app.route("/api/students/<int:student_id>", methods=["PUT"])
def api_update_student(student_id: int):
    data = request.get_json()
    name = (data.get("name") or "").strip()
    national_id = (data.get("national_id") or "").strip()
    phone = (data.get("phone") or "").strip() or None
    image_b64 = data.get("image")

    if not name or not national_id:
        return jsonify({"success": False, "message": "الاسم ورقم الهوية مطلوبان"}), 400

    encoding, photo_path = None, None
    if image_b64:
        frame = decode_base64_to_frame(image_b64)
        if frame is None:
            return jsonify({"success": False, "message": "الصورة الجديدة غير صالحة"}), 400
        try:
            encoding, _ = get_face_encoding_from_frame_strict(frame)
        except ValueError as e:
            return jsonify({"success": False, "message": str(e)}), 400
        photo_path = save_face_photo(frame)

    try:
        database.update_student(student_id, name, national_id, phone, encoding, photo_path)
    except database.DuplicateNationalIdError:
        return jsonify({"success": False, "message": "رقم الهوية/الوثيقة مستخدم لطالب آخر"}), 409

    return jsonify({"success": True, "message": "تم حفظ التعديلات بنجاح"})


@app.route("/api/students/<int:student_id>", methods=["DELETE"])
def api_delete_student(student_id: int):
    database.delete_student(student_id)
    return jsonify({"success": True, "message": "تم الحذف بنجاح"})


# --- API: weeks & lessons ---

@app.route("/api/week")
def api_week():
    date_param = request.args.get("date")
    if date_param:
        week = attendance_logic.get_week_for_date(date_param)
    else:
        offset = int(request.args.get("offset", 0))
        week = attendance_logic.get_week(offset)
    return jsonify(week)


@app.route("/api/lessons/<int:lesson_id>")
def api_lesson(lesson_id: int):
    lesson = database.get_lesson_by_id(lesson_id)
    if not lesson:
        return jsonify({"success": False, "message": "الدرس غير موجود"}), 404
    return jsonify({
        "lesson": lesson,
        "timing": attendance_logic.get_lesson_timing(lesson),
        "summary": attendance_logic.get_lesson_summary(lesson_id),
    })


@app.route("/api/lessons/<int:lesson_id>/roster")
def api_lesson_roster(lesson_id: int):
    return jsonify(attendance_logic.get_lesson_roster(lesson_id))


@app.route("/api/lessons/<int:lesson_id>/start", methods=["POST"])
def api_start_lesson(lesson_id: int):
    lesson = attendance_logic.start_lesson(lesson_id)
    return jsonify({"success": True, "lesson": lesson})


@app.route("/api/lessons/<int:lesson_id>/close", methods=["POST"])
def api_close_lesson(lesson_id: int):
    lesson = attendance_logic.close_lesson(lesson_id)
    return jsonify({"success": True, "lesson": lesson})


@app.route("/api/lessons/<int:lesson_id>/cancel", methods=["POST"])
def api_cancel_lesson(lesson_id: int):
    """Revert a mistakenly-started lesson back to not-started, discarding its attendance."""
    lesson = attendance_logic.cancel_lesson(lesson_id)
    return jsonify({"success": True, "lesson": lesson})


@app.route("/api/lessons/<int:lesson_id>/reopen", methods=["POST"])
def api_reopen_lesson(lesson_id: int):
    """Reopen a mistakenly-closed lesson, keeping real attendance but clearing auto-marked absences."""
    lesson = attendance_logic.reopen_lesson(lesson_id)
    return jsonify({"success": True, "lesson": lesson})


def _checkin_message(student_name: str, result: dict) -> str:
    """Build a user-facing message for a check-in result, noting duplicates."""
    if result["duplicate"]:
        return f"{student_name} مسجل مسبقاً بحالة: {result['status']}"
    return f"تم تسجيل {student_name} كـ {result['status']}"


@app.route("/api/lessons/<int:lesson_id>/checkin", methods=["POST"])
def api_checkin(lesson_id: int):
    """Manual check-in, used after searching for a student by name/ID/phone."""
    data = request.get_json()
    student_id = data.get("student_id")
    student = database.get_student_by_id(student_id)
    if not student:
        return jsonify({"success": False, "message": "الطالب غير موجود"}), 404

    try:
        result = attendance_logic.check_in_student(student_id, lesson_id, method="manual")
    except attendance_logic.LessonNotStartedError:
        return jsonify({"success": False, "message": "لم يبدأ الدرس بعد"}), 400

    return jsonify({"success": True, "message": _checkin_message(student["name"], result), "result": result})


@app.route("/api/lessons/<int:lesson_id>/checkin_by_national_id", methods=["POST"])
def api_checkin_by_national_id(lesson_id: int):
    """Fallback check-in by typing a national ID directly, without face recognition."""
    data = request.get_json()
    national_id = (data.get("national_id") or "").strip()
    student = database.get_student_by_national_id(national_id)
    if not student:
        return jsonify({"success": False, "message": "رقم الهوية غير مسجل بالنظام"}), 404

    try:
        result = attendance_logic.check_in_student(student["id"], lesson_id, method="id")
    except attendance_logic.LessonNotStartedError:
        return jsonify({"success": False, "message": "لم يبدأ الدرس بعد"}), 400

    return jsonify({
        "success": True,
        "message": _checkin_message(student["name"], result),
        "student": student,
        "result": result,
    })


@app.route("/api/lessons/<int:lesson_id>/apology", methods=["POST"])
def api_apology(lesson_id: int):
    """Excuse a student from a lesson, before or after it has happened."""
    data = request.get_json()
    student_id = data.get("student_id")

    try:
        record = attendance_logic.apply_apology(student_id, lesson_id)
    except database.DuplicateAttendanceError:
        return jsonify({"success": False, "message": "الطالب لديه حضور فعلي مسجل، لا يمكن تحويله لاعتذار"}), 409

    return jsonify({"success": True, "record": record})


# --- API: term-long apologies ---

@app.route("/api/term-apologies", methods=["GET"])
def api_get_term_apologies():
    return jsonify(database.get_term_apologies())


@app.route("/api/term-apologies", methods=["POST"])
def api_add_term_apology():
    data = request.get_json()
    student_id = data.get("student_id")
    start_date = data.get("start_date")
    end_date = data.get("end_date")

    if not student_id or not start_date or not end_date:
        return jsonify({"success": False, "message": "حدد الطالب وتاريخ البداية والنهاية"}), 400
    if end_date < start_date:
        return jsonify({"success": False, "message": "تاريخ النهاية يجب أن يكون بعد تاريخ البداية"}), 400

    record = database.add_term_apology(student_id, start_date, end_date, data.get("reason"))
    return jsonify({"success": True, "record": record})


@app.route("/api/term-apologies/<int:apology_id>", methods=["DELETE"])
def api_delete_term_apology(apology_id: int):
    database.delete_term_apology(apology_id)
    return jsonify({"success": True})


@app.route("/api/lessons/<int:lesson_id>/recognize_live", methods=["POST"])
def api_recognize_live(lesson_id: int):
    """Recognize every face in the frame and check in each matched student.

    Faces that don't match a known student are reported as unrecognized
    without any database write; faces that do match are passed through the
    same check-in logic used by manual and ID-based attendance, so the
    on-time/late rule and duplicate protection apply identically regardless
    of how the student was recognized.
    """
    data = request.get_json()
    image_b64 = data.get("image")

    frame = decode_base64_to_frame(image_b64)
    if frame is None:
        return jsonify({"success": True, "faces": []})

    detections = get_all_face_encodings_from_frame(frame)
    known_students = database.get_cached_students()
    frame_size = {"width": frame.shape[1], "height": frame.shape[0]}

    faces = []
    for encoding, face_location in detections:
        top, right, bottom, left = face_location
        box = {"top": top, "right": right, "bottom": bottom, "left": left}

        match, distance = find_match(encoding, known_students)
        if match is None:
            faces.append({
                "known": False, "box": box, "label": "غير مسجل", "message": "⚠️ شخص غير مسجل",
            })
            continue

        try:
            result = attendance_logic.check_in_student(match["id"], lesson_id, method="face")
        except attendance_logic.LessonNotStartedError:
            faces.append({
                "known": True, "box": box, "label": match["name"],
                "status": None, "message": f"{match['name']}: لم يبدأ الدرس بعد",
            })
            continue

        faces.append({
            "known": True,
            "box": box,
            "label": match["name"],
            "status": result["status"],
            "message": _checkin_message(match["name"], result),
            "distance": round(distance, 3),
        })

    return jsonify({"success": True, "frame_size": frame_size, "faces": faces})


# --- API: camera settings ---

@app.route("/api/camera-settings", methods=["GET"])
def api_get_camera_settings():
    return jsonify(database.get_active_camera() or {})


@app.route("/api/camera-settings", methods=["POST"])
def api_save_camera_settings():
    data = request.get_json()
    camera = database.save_camera_settings(
        name=data.get("name", "الكاميرا الرئيسية"),
        ip_address=data["ip_address"],
        port=int(data.get("port") or 554),
        username=data.get("username"),
        password=data.get("password"),
        protocol=data.get("protocol", "rtsp"),
        stream_path=data.get("stream_path", ""),
    )
    return jsonify({"success": True, "camera": camera})


@app.route("/api/camera-settings/test", methods=["POST"])
def api_test_camera():
    camera = database.get_active_camera()
    if not camera:
        return jsonify({"success": False, "message": "لم يتم إعداد الكاميرا بعد", "status": "unknown"}), 400

    result = camera_manager.test_connection(camera)
    return jsonify({"success": result["status"] == "connected", **result})


# --- API: live CCTV stream ---

@app.route("/api/camera/stream/start/<int:lesson_id>", methods=["POST"])
def api_start_camera_stream(lesson_id: int):
    """Start (or retarget) the shared CCTV stream to check faces into this lesson."""
    camera_stream.start_stream(lesson_id)
    return jsonify({"success": True})


@app.route("/api/camera/stream/stop", methods=["POST"])
def api_stop_camera_stream():
    camera_stream.stop_stream()
    return jsonify({"success": True})


@app.route("/api/camera/stream/status")
def api_camera_stream_status():
    return jsonify(camera_stream.get_status())


@app.route("/video_feed")
def video_feed():
    """MJPEG endpoint: browsers render this directly with a plain <img> tag.

    Each part is a full JPEG frame; the browser keeps redrawing the <img>
    as new parts arrive, which is enough to look like live video without
    needing WebRTC or any video-specific browser API.
    """
    def generate():
        while True:
            frame = camera_stream.get_latest_jpeg()
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.05)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# --- API: general settings ---

@app.route("/api/settings/late-threshold", methods=["GET"])
def api_get_late_threshold():
    return jsonify({"minutes": database.get_late_threshold_minutes()})


@app.route("/api/settings/late-threshold", methods=["POST"])
def api_set_late_threshold():
    data = request.get_json()
    minutes = int(data.get("minutes", 30))
    database.set_setting("late_threshold_minutes", str(minutes))
    return jsonify({"success": True, "minutes": minutes})


# --- API: Excel export ---

@app.route("/api/export/excel")
def api_export_excel():
    """Export a single week's attendance (both lessons) as a real .xlsx file."""
    date_param = request.args.get("date")
    status_filter = request.args.get("status") or None
    if date_param:
        week = attendance_logic.get_week_for_date(date_param)
    else:
        offset = int(request.args.get("offset", 0))
        week = attendance_logic.get_week(offset)
    buffer = excel_export.export_week_to_bytes(week, status_filter=status_filter)

    return send_file(
        buffer,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"attendance_week_{week['start_date']}.xlsx",
    )


@app.route("/api/export/excel/range")
def api_export_excel_range():
    """Export every lesson within an arbitrary date range as a real .xlsx file."""
    start_date = request.args.get("start")
    end_date = request.args.get("end")
    status_filter = request.args.get("status") or None
    if not start_date or not end_date:
        return jsonify({"success": False, "message": "الرجاء تحديد تاريخ البداية والنهاية"}), 400

    buffer = excel_export.export_range_to_bytes(start_date, end_date, status_filter=status_filter)
    return send_file(
        buffer,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"attendance_{start_date}_to_{end_date}.xlsx",
    )


if __name__ == "__main__":
    app.run(debug=True, threaded=True)