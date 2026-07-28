import base64
import os

import numpy as np
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set in the .env file.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# In-memory cache of known persons, loaded once at startup so recognition
# never has to hit the database on the hot path.
known_faces_cache: list[dict] = []


# --- Encoding serialization ---

# Serializes a face encoding to base64 for storage as a text column.
def encode_to_text(encoding: np.ndarray) -> str:
    """Encode a NumPy face encoding as a base64 string."""
    return base64.b64encode(encoding.tobytes()).decode("utf-8")


# Deserializes a stored base64 string back into a face encoding.
def decode_from_text(encoded_str: str) -> np.ndarray:
    """Decode a base64 string back into a NumPy face encoding."""
    raw_bytes = base64.b64decode(encoded_str)
    return np.frombuffer(raw_bytes, dtype=np.float64)


# --- In-memory cache ---

# Loads all persons from Supabase into the in-memory cache at startup.
def load_cache_from_db() -> list[dict]:
    """Refresh the in-memory persons cache from the database."""
    global known_faces_cache
    result = supabase.table("persons").select("*").execute()

    known_faces_cache = [
        {
            "id": row["id"],
            "name": row["name"],
            "encoding": decode_from_text(row["face_encoding"]),
        }
        for row in result.data
    ]

    print(f"Loaded {len(known_faces_cache)} persons into memory.")
    return known_faces_cache


# Returns the current in-memory persons cache used during recognition.
def get_cached_persons() -> list[dict]:
    """Return the cached list of known persons."""
    return known_faces_cache


# --- Write operations ---

# Inserts a new person into Supabase and appends them to the in-memory cache.
def add_person(
    name: str,
    encoding: np.ndarray,
    photo_path: str = None,
    phone: str = None,
) -> dict:
    """Register a new person and update the in-memory cache."""
    encoded_str = encode_to_text(encoding)

    result = supabase.table("persons").insert({
        "name": name,
        "face_encoding": encoded_str,
        "photo_path": photo_path,
        "phone": phone,
    }).execute()

    new_person = result.data[0]

    known_faces_cache.append({
        "id": new_person["id"],
        "name": new_person["name"],
        "encoding": encoding,
    })

    return new_person


# Removes a person from Supabase and evicts them from the in-memory cache.
def delete_person(person_id: int) -> None:
    """Delete a person from the database and the in-memory cache."""
    global known_faces_cache
    supabase.table("persons").delete().eq("id", person_id).execute()
    known_faces_cache = [p for p in known_faces_cache if p["id"] != person_id]


# Records a check-in/check-out event for a person.
def log_attendance(
    person_id: int,
    person_name: str,
    confidence: float,
    status: str = "حضور",
) -> dict:
    """Insert an attendance record for a recognized person."""
    result = supabase.table("attendance").insert({
        "person_id": person_id,
        "person_name": person_name,
        "confidence": float(confidence),
        "status": status,
    }).execute()
    return result.data[0]


# --- Read operations (reporting) ---

# Fetches recent attendance records for the reporting dashboard.
def get_attendance_records(limit: int = 200) -> list[dict]:
    """Return the most recent attendance records, newest first."""
    result = (
        supabase.table("attendance")
        .select("*")
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data


# Fetches all registered persons for the management page.
def get_all_persons_from_db() -> list[dict]:
    """Return all registered persons, newest first."""
    result = (
        supabase.table("persons")
        .select("id, name, phone, photo_path, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return result.data