import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).parent / "attendance.db"

# A single sqlite3.Connection is not safe for concurrent use across threads,
# even with check_same_thread=False (that flag only disables the safety
# check, it doesn't make concurrent access correct). Flask's threaded dev
# server plus the frontend's overlapping polling (recognize_live, lesson
# info, roster all firing within the same second) caused real concurrent
# access, which surfaced as intermittent "bad parameter or other API
# misuse" errors. Every single database call -- reads included, not just
# writes -- is now serialized through this one lock.
_db_lock = threading.Lock()
_connection = sqlite3.connect(DB_PATH, check_same_thread=False)
_connection.row_factory = sqlite3.Row
_connection.execute("PRAGMA foreign_keys = ON")


# --- Schema ---

def init_db() -> None:
    """Create all tables if they don't exist yet. Safe to call on every startup."""
    with _db_lock, _connection:
        _connection.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                phone TEXT,
                national_id TEXT NOT NULL UNIQUE,
                face_encoding BLOB NOT NULL,
                photo_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        _connection.execute("""
            CREATE TABLE IF NOT EXISTS weeks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_date TEXT NOT NULL UNIQUE
            )
        """)

        _connection.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week_id INTEGER NOT NULL REFERENCES weeks(id) ON DELETE CASCADE,
                day_type TEXT NOT NULL CHECK (day_type IN ('sunday', 'thursday')),
                lesson_date TEXT NOT NULL,
                late_threshold_minutes INTEGER,
                started_at TIMESTAMP,
                closed_at TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'not_started'
                    CHECK (status IN ('not_started', 'in_progress', 'closed')),
                UNIQUE (week_id, day_type)
            )
        """)

        _connection.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                lesson_id INTEGER NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
                status TEXT NOT NULL CHECK (status IN ('حاضر', 'متأخر', 'معتذر', 'غائب')),
                check_in_time TIMESTAMP,
                method TEXT CHECK (method IN ('face', 'manual', 'id')),
                UNIQUE (student_id, lesson_id)
            )
        """)

        _connection.execute("""
            CREATE TABLE IF NOT EXISTS camera_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                ip_address TEXT NOT NULL,
                port INTEGER NOT NULL DEFAULT 554,
                username TEXT,
                password TEXT,
                protocol TEXT NOT NULL DEFAULT 'rtsp',
                stream_path TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                last_status TEXT NOT NULL DEFAULT 'unknown',
                last_checked_at TIMESTAMP
            )
        """)

        _connection.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        _connection.execute("""
            INSERT OR IGNORE INTO app_settings (key, value)
            VALUES ('late_threshold_minutes', '30')
        """)

        _connection.execute("""
            CREATE TABLE IF NOT EXISTS term_apologies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                reason TEXT
            )
        """)


init_db()

student_cache: list[dict] = []


def load_student_cache() -> list[dict]:
    """Refresh the in-memory students cache from the database."""
    global student_cache
    with _db_lock:
        rows = _connection.execute(
            "SELECT id, name, national_id, face_encoding FROM students"
        ).fetchall()

    student_cache = [
        {
            "id": row["id"],
            "name": row["name"],
            "national_id": row["national_id"],
            "encoding": np.frombuffer(row["face_encoding"], dtype=np.float64),
        }
        for row in rows
    ]

    print(f"Loaded {len(student_cache)} students into memory.")
    return student_cache


def get_cached_students() -> list[dict]:
    """Return the cached list of students used during recognition."""
    return student_cache


# --- Students ---

class DuplicateNationalIdError(Exception):
    """Raised when registering a student with a national ID already in use."""


def add_student(
    name: str,
    national_id: str,
    encoding: np.ndarray,
    photo_path: str = None,
    phone: str = None,
) -> dict:
    """Register a new student and update the in-memory cache."""
    encoding_bytes = encoding.tobytes()

    try:
        with _db_lock, _connection:
            cursor = _connection.execute(
                """INSERT INTO students (name, phone, national_id, face_encoding, photo_path)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, phone, national_id, encoding_bytes, photo_path),
            )
            student_id = cursor.lastrowid
    except sqlite3.IntegrityError as e:
        if "national_id" in str(e):
            raise DuplicateNationalIdError(
                f"National ID '{national_id}' is already registered."
            ) from e
        raise

    student_cache.append({
        "id": student_id,
        "name": name,
        "national_id": national_id,
        "encoding": encoding,
    })

    return {"id": student_id, "name": name, "national_id": national_id, "phone": phone}


def update_student(
    student_id: int,
    name: str,
    national_id: str,
    phone: str = None,
    encoding: np.ndarray = None,
    photo_path: str = None,
) -> None:
    """Update a student's details, optionally including a new face encoding."""
    try:
        with _db_lock, _connection:
            if encoding is not None:
                _connection.execute(
                    """UPDATE students
                       SET name = ?, phone = ?, national_id = ?,
                           face_encoding = ?, photo_path = COALESCE(?, photo_path)
                       WHERE id = ?""",
                    (name, phone, national_id, encoding.tobytes(), photo_path, student_id),
                )
            else:
                _connection.execute(
                    """UPDATE students
                       SET name = ?, phone = ?, national_id = ?
                       WHERE id = ?""",
                    (name, phone, national_id, student_id),
                )
    except sqlite3.IntegrityError as e:
        if "national_id" in str(e):
            raise DuplicateNationalIdError(
                f"National ID '{national_id}' is already registered to another student."
            ) from e
        raise

    load_student_cache()


def delete_student(student_id: int) -> None:
    """Delete a student and their attendance history, and refresh the cache."""
    with _db_lock, _connection:
        _connection.execute("DELETE FROM students WHERE id = ?", (student_id,))
    load_student_cache()


def get_student_by_id(student_id: int) -> dict | None:
    """Return a single student's full record, or None if not found."""
    with _db_lock:
        row = _connection.execute(
            "SELECT id, name, phone, national_id, photo_path, created_at FROM students WHERE id = ?",
            (student_id,),
        ).fetchone()
    return dict(row) if row else None


def get_student_by_national_id(national_id: str) -> dict | None:
    """Return a student by their exact national ID, or None if not found."""
    with _db_lock:
        row = _connection.execute(
            "SELECT id, name, phone, national_id, photo_path, created_at FROM students WHERE national_id = ?",
            (national_id,),
        ).fetchone()
    return dict(row) if row else None


def search_students(query: str) -> list[dict]:
    """Search students by partial match on name, national ID, or phone."""
    like_query = f"%{query}%"
    with _db_lock:
        rows = _connection.execute(
            """SELECT id, name, phone, national_id, photo_path, created_at
               FROM students
               WHERE name LIKE ? OR national_id LIKE ? OR phone LIKE ?
               ORDER BY name""",
            (like_query, like_query, like_query),
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_students() -> list[dict]:
    """Return all registered students, newest first."""
    with _db_lock:
        rows = _connection.execute(
            "SELECT id, name, phone, national_id, photo_path, created_at FROM students ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


# --- Weeks & lessons ---

def get_or_create_week(start_date: str) -> dict:
    """Return the week starting on start_date (ISO YYYY-MM-DD), creating it if needed."""
    with _db_lock, _connection:
        _connection.execute(
            "INSERT OR IGNORE INTO weeks (start_date) VALUES (?)", (start_date,)
        )
        row = _connection.execute(
            "SELECT id, start_date FROM weeks WHERE start_date = ?", (start_date,)
        ).fetchone()
    return dict(row)


def list_weeks() -> list[dict]:
    """Return all weeks, most recent first."""
    with _db_lock:
        rows = _connection.execute(
            "SELECT id, start_date FROM weeks ORDER BY start_date DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_or_create_lesson(week_id: int, day_type: str, lesson_date: str) -> dict:
    """Return the lesson for a given week/day_type, creating it if needed."""
    with _db_lock, _connection:
        _connection.execute(
            """INSERT OR IGNORE INTO lessons (week_id, day_type, lesson_date)
               VALUES (?, ?, ?)""",
            (week_id, day_type, lesson_date),
        )
        row = _connection.execute(
            "SELECT * FROM lessons WHERE week_id = ? AND day_type = ?",
            (week_id, day_type),
        ).fetchone()
    return dict(row)


def get_lessons_in_range(start_date: str, end_date: str) -> list[dict]:
    """Return every lesson with a date within [start_date, end_date]."""
    with _db_lock:
        rows = _connection.execute(
            """SELECT * FROM lessons
               WHERE lesson_date BETWEEN ? AND ?
               ORDER BY lesson_date""",
            (start_date, end_date),
        ).fetchall()
    return [dict(row) for row in rows]


def get_lesson_by_id(lesson_id: int) -> dict | None:
    """Return a single lesson's record, or None if not found."""
    with _db_lock:
        row = _connection.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
    return dict(row) if row else None


def start_lesson(lesson_id: int, late_threshold_minutes: int) -> None:
    """Mark a lesson as started now, snapshotting the current late threshold."""
    with _db_lock, _connection:
        _connection.execute(
            """UPDATE lessons
               SET status = 'in_progress',
                   started_at = ?,
                   late_threshold_minutes = ?
               WHERE id = ?""",
            (datetime.now().isoformat(), late_threshold_minutes, lesson_id),
        )


def reset_lesson(lesson_id: int) -> None:
    """Cancel an in-progress lesson, reverting it to not-started."""
    with _db_lock, _connection:
        _connection.execute("DELETE FROM attendance WHERE lesson_id = ?", (lesson_id,))
        _connection.execute(
            """UPDATE lessons
               SET status = 'not_started', started_at = NULL,
                   closed_at = NULL, late_threshold_minutes = NULL
               WHERE id = ?""",
            (lesson_id,),
        )


def close_lesson(lesson_id: int) -> None:
    """Mark a lesson as closed, after which absences can be determined."""
    with _db_lock, _connection:
        _connection.execute(
            "UPDATE lessons SET status = 'closed', closed_at = ? WHERE id = ?",
            (datetime.now().isoformat(), lesson_id),
        )


def reopen_lesson(lesson_id: int) -> None:
    """Reopen a lesson that was closed by mistake, preserving real attendance."""
    with _db_lock, _connection:
        _connection.execute(
            "DELETE FROM attendance WHERE lesson_id = ? AND status = 'غائب'", (lesson_id,)
        )
        _connection.execute(
            "UPDATE lessons SET status = 'in_progress', closed_at = NULL WHERE id = ?",
            (lesson_id,),
        )


# --- Attendance ---

class DuplicateAttendanceError(Exception):
    """Raised when a student already has an attendance record for a lesson."""


def record_attendance(
    student_id: int,
    lesson_id: int,
    status: str,
    method: str = None,
) -> dict:
    """Insert an attendance/apology record for a student in a lesson."""
    try:
        with _db_lock, _connection:
            cursor = _connection.execute(
                """INSERT INTO attendance (student_id, lesson_id, status, check_in_time, method)
                   VALUES (?, ?, ?, ?, ?)""",
                (student_id, lesson_id, status, datetime.now().isoformat(), method),
            )
            record_id = cursor.lastrowid
    except sqlite3.IntegrityError as e:
        raise DuplicateAttendanceError(
            f"Student {student_id} already has an attendance record for lesson {lesson_id}."
        ) from e

    return {"id": record_id, "student_id": student_id, "lesson_id": lesson_id, "status": status}


def set_or_update_apology(student_id: int, lesson_id: int) -> dict:
    """Mark a student as excused for a lesson, before or after it has happened."""
    existing = get_attendance_record(student_id, lesson_id)

    if existing is None:
        with _db_lock, _connection:
            cursor = _connection.execute(
                """INSERT INTO attendance (student_id, lesson_id, status, method)
                   VALUES (?, ?, 'معتذر', NULL)""",
                (student_id, lesson_id),
            )
            record_id = cursor.lastrowid
        return {"id": record_id, "student_id": student_id, "lesson_id": lesson_id, "status": "معتذر"}

    if existing["status"] == "غائب":
        with _db_lock, _connection:
            _connection.execute(
                "UPDATE attendance SET status = 'معتذر' WHERE id = ?", (existing["id"],)
            )
        return {**existing, "status": "معتذر"}

    if existing["status"] == "معتذر":
        return existing

    raise DuplicateAttendanceError(
        f"Student {student_id} already has a recorded attendance ({existing['status']}) for lesson {lesson_id}."
    )


def mark_absent(student_id: int, lesson_id: int) -> None:
    """Record a student as absent, with no check-in time or method."""
    with _db_lock, _connection:
        _connection.execute(
            """INSERT OR IGNORE INTO attendance (student_id, lesson_id, status, check_in_time, method)
               VALUES (?, ?, 'غائب', NULL, NULL)""",
            (student_id, lesson_id),
        )


def get_attendance_record(student_id: int, lesson_id: int) -> dict | None:
    """Return a student's attendance record for a lesson, if one exists."""
    with _db_lock:
        row = _connection.execute(
            "SELECT * FROM attendance WHERE student_id = ? AND lesson_id = ?",
            (student_id, lesson_id),
        ).fetchone()
    return dict(row) if row else None


def get_lesson_attendance(lesson_id: int) -> list[dict]:
    """Return every recorded attendance row for a lesson, joined with student names."""
    with _db_lock:
        rows = _connection.execute(
            """SELECT a.*, s.name AS student_name, s.national_id
               FROM attendance a
               JOIN students s ON s.id = a.student_id
               WHERE a.lesson_id = ?
               ORDER BY a.check_in_time""",
            (lesson_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_student_history(student_id: int) -> list[dict]:
    """Return a student's full attendance history across all lessons/weeks."""
    with _db_lock:
        rows = _connection.execute(
            """SELECT a.status, a.check_in_time, a.method,
                      l.day_type, l.lesson_date, w.start_date AS week_start
               FROM attendance a
               JOIN lessons l ON l.id = a.lesson_id
               JOIN weeks w ON w.id = l.week_id
               WHERE a.student_id = ?
               ORDER BY w.start_date DESC, l.day_type""",
            (student_id,),
        ).fetchall()
    return [dict(row) for row in rows]


# --- Camera settings ---

def get_active_camera() -> dict | None:
    """Return the currently active camera's settings, if one is configured."""
    with _db_lock:
        row = _connection.execute(
            "SELECT * FROM camera_settings WHERE is_active = 1 LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def save_camera_settings(
    name: str,
    ip_address: str,
    port: int,
    username: str,
    password: str,
    protocol: str,
    stream_path: str,
) -> dict:
    """Create or update the single active camera's connection settings."""
    existing = get_active_camera()

    with _db_lock, _connection:
        if existing:
            _connection.execute(
                """UPDATE camera_settings
                   SET name = ?, ip_address = ?, port = ?, username = ?,
                       password = ?, protocol = ?, stream_path = ?
                   WHERE id = ?""",
                (name, ip_address, port, username, password, protocol, stream_path, existing["id"]),
            )
            camera_id = existing["id"]
        else:
            cursor = _connection.execute(
                """INSERT INTO camera_settings
                   (name, ip_address, port, username, password, protocol, stream_path, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
                (name, ip_address, port, username, password, protocol, stream_path),
            )
            camera_id = cursor.lastrowid

    return get_active_camera() or {"id": camera_id}


def update_camera_status(camera_id: int, status: str) -> None:
    """Record the result of the most recent connection test for a camera."""
    with _db_lock, _connection:
        _connection.execute(
            "UPDATE camera_settings SET last_status = ?, last_checked_at = ? WHERE id = ?",
            (status, datetime.now().isoformat(), camera_id),
        )


# --- App settings ---

def get_setting(key: str, default: str = None) -> str | None:
    """Return a setting's value, or default if not set."""
    with _db_lock:
        row = _connection.execute(
            "SELECT value FROM app_settings WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    """Create or update a setting's value."""
    with _db_lock, _connection:
        _connection.execute(
            """INSERT INTO app_settings (key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (key, value),
        )


def get_late_threshold_minutes() -> int:
    """Return the currently configured default late threshold, in minutes."""
    return int(get_setting("late_threshold_minutes", "30"))


# --- Term-long apologies ---

def add_term_apology(student_id: int, start_date: str, end_date: str, reason: str = None) -> dict:
    """Excuse a student for every lesson falling within a date range."""
    with _db_lock, _connection:
        cursor = _connection.execute(
            """INSERT INTO term_apologies (student_id, start_date, end_date, reason)
               VALUES (?, ?, ?, ?)""",
            (student_id, start_date, end_date, reason),
        )
        apology_id = cursor.lastrowid
    return {"id": apology_id, "student_id": student_id, "start_date": start_date, "end_date": end_date}


def get_term_apologies() -> list[dict]:
    """Return every term-long apology, joined with the student's name."""
    with _db_lock:
        rows = _connection.execute(
            """SELECT ta.*, s.name AS student_name, s.national_id
               FROM term_apologies ta
               JOIN students s ON s.id = ta.student_id
               ORDER BY ta.start_date DESC"""
        ).fetchall()
    return [dict(row) for row in rows]


def delete_term_apology(apology_id: int) -> None:
    """Remove a term-long apology."""
    with _db_lock, _connection:
        _connection.execute("DELETE FROM term_apologies WHERE id = ?", (apology_id,))


def get_active_term_apology(student_id: int, lesson_date: str) -> dict | None:
    """Return a student's term-long apology covering lesson_date, if any."""
    with _db_lock:
        row = _connection.execute(
            """SELECT * FROM term_apologies
               WHERE student_id = ? AND start_date <= ? AND end_date >= ?
               LIMIT 1""",
            (student_id, lesson_date, lesson_date),
        ).fetchone()
    return dict(row) if row else None