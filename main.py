from src.api import DeepPerson

dp = DeepPerson(model_name="resnet50_circle_dg", device="cuda", detector_backend="yolo")

# 1. Generate embeddings
print("=== Represent ===")
result1 = dp.represent("wei_song.jpeg")
print(f"Subjects detected: {len(result1['subjects'])}")
print(f"Embedding shape: {result1['subjects'][0]['embedding'].shape}")

result2 = dp.represent("back.jpg")
print(f"Subjects detected: {len(result2['subjects'])}")
print(f"Embedding shape: {result2['subjects'][0]['embedding'].shape}")

# 2. Verify (expected: different persons)
print("\n=== Verify ===")
verify_result = dp.verify("wei_song.jpeg", "back.jpg", distance_metric="cosine")
print(f"Same person: {verify_result['verified']}")
print(f"Distance: {verify_result['distance']:.4f}")
print(f"Threshold: {verify_result['threshold']}")

# 3. Build gallery
print("\n=== Build Gallery ===")
gallery_result = dp.build_gallery(
    img_paths=["wei_song.jpeg", "back.jpg"],
    subject_ids=["wei_song", "back"],
    gallery_path="./gallery",
    gallery_name="demo"
)
print(f"Gallery entries: {gallery_result['processed']}")

# 4. Search gallery
print("\n=== Find ===")
find_result = dp.find("back.jpg", "./gallery", gallery_name="demo", top_k=1)
print(f"Top match: {find_result['matches'][0]['subject_id']}")
print(f"Distance: {find_result['matches'][0]['distance']:.4f}")