from datetime import datetime, timedelta

import database

# --- Week/lesson resolution ---

# Python's weekday(): Monday=0 ... Sunday=6. Used to walk back/forward to
# the Sunday that starts a given calendar week.
_SUNDAY_WEEKDAY = 6


def _sunday_of(date: datetime) -> datetime:
    """Return the Sunday that starts the calendar week containing date."""
    days_since_sunday = (date.weekday() - _SUNDAY_WEEKDAY) % 7
    return date - timedelta(days=days_since_sunday)


def get_week(offset: int = 0) -> dict:
    """Return the week at the given offset from the current week (0 = current,
    -1 = previous, +1 = next), creating its Sunday/Thursday lessons if needed.
    """
    sunday = _sunday_of(datetime.now()) + timedelta(weeks=offset)
    return _build_week(sunday)


def get_week_for_date(date_string: str) -> dict:
    """Return the week (Sunday-Thursday) containing the given date (YYYY-MM-DD).

    Used by the calendar date picker, so an admin can jump directly to any
    date's week instead of clicking prev/next repeatedly.
    """
    target_date = datetime.strptime(date_string, "%Y-%m-%d")
    sunday = _sunday_of(target_date)
    return _build_week(sunday)


def _build_week(sunday: datetime) -> dict:
    """Ensure a week and its two lessons exist in the database, and return them."""
    thursday = sunday + timedelta(days=4)

    week = database.get_or_create_week(sunday.strftime("%Y-%m-%d"))
    sunday_lesson = database.get_or_create_lesson(week["id"], "sunday", sunday.strftime("%Y-%m-%d"))
    thursday_lesson = database.get_or_create_lesson(week["id"], "thursday", thursday.strftime("%Y-%m-%d"))

    return {
        "id": week["id"],
        "start_date": week["start_date"],
        "sunday_lesson": sunday_lesson,
        "thursday_lesson": thursday_lesson,
    }


# --- Session lifecycle ---

def start_lesson(lesson_id: int) -> dict:
    """Start a lesson now, snapshotting the currently configured late threshold."""
    late_threshold = database.get_late_threshold_minutes()
    database.start_lesson(lesson_id, late_threshold)
    return database.get_lesson_by_id(lesson_id)


def cancel_lesson(lesson_id: int) -> dict:
    """Cancel a lesson that was started by mistake, reverting it to not-started.

    Any attendance recorded during the cancelled session is discarded, so
    the lesson can be started again cleanly -- e.g. if "Start" was pressed
    at the wrong time.
    """
    database.reset_lesson(lesson_id)
    return database.get_lesson_by_id(lesson_id)


def close_lesson(lesson_id: int) -> dict:
    """Close a lesson and settle every student without a record.

    Absence is only ever determined here, at closing time, so a student is
    never shown as absent while they might still walk in before the lesson
    officially ends. Students covered by a term-long apology are marked
    excused instead of absent, since they were never expected to attend.
    """
    lesson = database.get_lesson_by_id(lesson_id)

    for student in database.get_all_students():
        if database.get_attendance_record(student["id"], lesson_id) is not None:
            continue

        term_apology = database.get_active_term_apology(student["id"], lesson["lesson_date"])
        if term_apology:
            database.set_or_update_apology(student["id"], lesson_id)
        else:
            database.mark_absent(student["id"], lesson_id)

    database.close_lesson(lesson_id)
    return database.get_lesson_by_id(lesson_id)


def reopen_lesson(lesson_id: int) -> dict:
    """Reopen a lesson that was closed by mistake, preserving real attendance.

    Only the automatically-generated absent rows are cleared; recognition,
    manual, and ID check-ins can then resume normally.
    """
    database.reopen_lesson(lesson_id)
    return database.get_lesson_by_id(lesson_id)


# --- Timing ---

def get_lesson_timing(lesson: dict) -> dict:
    """Compute the current timing state of a lesson for display.

    Returns a dict with the lesson's phase, elapsed/remaining durations,
    and whether the late period has started -- all derived live from
    started_at, never from wall-clock time alone, since lateness only
    ever starts counting from the moment "Start" was pressed.
    """
    if lesson["status"] == "not_started":
        return {"phase": "not_started", "elapsed_seconds": None, "remaining_seconds": None}

    if lesson["status"] == "closed":
        return {"phase": "closed", "elapsed_seconds": None, "remaining_seconds": None}

    started_at = datetime.fromisoformat(lesson["started_at"])
    threshold = timedelta(minutes=lesson["late_threshold_minutes"])
    elapsed = datetime.now() - started_at
    remaining = threshold - elapsed

    if remaining.total_seconds() > 0:
        return {
            "phase": "on_time",
            "elapsed_seconds": int(elapsed.total_seconds()),
            "remaining_seconds": int(remaining.total_seconds()),
        }

    return {
        "phase": "late_period",
        "elapsed_seconds": int(elapsed.total_seconds()),
        "remaining_seconds": 0,
    }


def _determine_checkin_status(lesson: dict) -> str:
    """Return 'حاضر' or 'متأخر' for a check-in happening right now."""
    timing = get_lesson_timing(lesson)
    return "متأخر" if timing["phase"] == "late_period" else "حاضر"


# --- Check-in ---

class LessonNotStartedError(Exception):
    """Raised when attempting to check in before the lesson has been started."""


def check_in_student(student_id: int, lesson_id: int, method: str) -> dict:
    """Record a student's attendance for a lesson, applying the on-time/late rule.

    If the student already has a record for this lesson (present, late, or
    excused), the existing record is returned as-is rather than raising,
    so recognizing the same face twice -- or recognizing a face after a
    manual/ID check-in -- never creates a duplicate or overwrites an
    apology with a plain "present".
    """
    lesson = database.get_lesson_by_id(lesson_id)

    existing = database.get_attendance_record(student_id, lesson_id)
    if existing is not None:
        return {**existing, "duplicate": True}

    if lesson["status"] != "in_progress":
        raise LessonNotStartedError("Attendance can only be recorded while the lesson is in progress.")

    status = _determine_checkin_status(lesson)
    record = database.record_attendance(student_id, lesson_id, status, method=method)
    return {**record, "duplicate": False}


def apply_apology(student_id: int, lesson_id: int) -> dict:
    """Mark a student as excused for a lesson, before or after it has happened.

    Works whether the apology is set in advance or submitted after the
    student was already marked absent; a genuine present/late record is
    never overwritten.

    Raises database.DuplicateAttendanceError if the student already has a
    real attendance record (present/late) for this lesson.
    """
    return database.set_or_update_apology(student_id, lesson_id)


# --- Reporting ---

def get_lesson_roster(lesson_id: int) -> list[dict]:
    """Return every student's status for a lesson, including those pending.

    Students with no attendance record yet are reported as 'لم يسجل بعد'
    while the lesson is still open (never as absent), and only backfilled
    to 'غائب' once the lesson has actually been closed.
    """
    lesson = database.get_lesson_by_id(lesson_id)
    records_by_student = {
        record["student_id"]: record for record in database.get_lesson_attendance(lesson_id)
    }

    roster = []
    for student in database.get_all_students():
        record = records_by_student.get(student["id"])
        if record is not None:
            roster.append({
                "student_id": student["id"],
                "name": student["name"],
                "national_id": student["national_id"],
                "phone": student.get("phone"),
                "status": record["status"],
                "check_in_time": record["check_in_time"],
                "method": record["method"],
            })
        else:
            if lesson["status"] == "closed":
                pending_status = "غائب"
            else:
                term_apology = database.get_active_term_apology(student["id"], lesson["lesson_date"])
                pending_status = "معتذر" if term_apology else "لم يسجل بعد"
            roster.append({
                "student_id": student["id"],
                "name": student["name"],
                "national_id": student["national_id"],
                "phone": student.get("phone"),
                "status": pending_status,
                "check_in_time": None,
                "method": None,
            })

    return roster


def get_lesson_summary(lesson_id: int) -> dict:
    """Return attendance counts by status for a lesson, for dashboard display."""
    roster = get_lesson_roster(lesson_id)
    counts = {"حاضر": 0, "متأخر": 0, "معتذر": 0, "غائب": 0, "لم يسجل بعد": 0}
    for entry in roster:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    return counts