from face_engine import find_match, get_face_encoding_from_path

# Two photos of the same person and one photo of a different person

encoding1 = get_face_encoding_from_path("known_faces//person2_a.jpg")
encoding2 = get_face_encoding_from_path("known_faces//person2_b.jpg")
encoding3 = get_face_encoding_from_path("known_faces//person1.png")

# Build a "known persons" list containing a single person.
known_persons = [
    {"id": 1, "name": "Person One", "encoding": encoding1}
]

# Test 1: same person.
match, distance = find_match(encoding2, known_persons)
print(f"Same-person test -> distance: {distance:.3f}, result: {match['name'] if match else 'unknown'}")

# Test 2: different person.
match, distance = find_match(encoding3, known_persons)
print(f"Different-person test -> distance: {distance:.3f}, result: {match['name'] if match else 'unknown'}")