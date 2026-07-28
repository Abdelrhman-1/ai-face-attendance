from database import add_person, get_attendance_records, load_cache_from_db, log_attendance
from face_engine import get_face_encoding_from_path

# 1. Load the cache (expected to be 0 since the table is empty initially).
persons = load_cache_from_db()
print(f"Persons in cache: {len(persons)}")

# 2. Add a real new person.
encoding = get_face_encoding_from_path("known_faces/person2_b.jpg")
new_person = add_person("Ahmed", encoding, "known_faces/person1_a.jpg")
print(f"Added: {new_person}")

# 3. Confirm the cache updated automatically without a full reload.
print(f"Persons in cache after add: {len(get_cached_persons() if False else persons)}")

# 4. Log a test attendance record for the same person.
attendance_record = log_attendance(
    person_id=new_person["id"],
    person_name=new_person["name"],
    confidence=0.30,
)
print(f"Attendance logged: {attendance_record}")

# 5. Fetch the latest records and confirm it shows up.
records = get_attendance_records(limit=5)
print(f"Latest records: {records}")