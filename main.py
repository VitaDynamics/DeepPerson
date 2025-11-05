from pathlib import Path

from src.api import DeepPerson


SAMPLE_IMAGE = Path("back.jpg")
GALLERY_STORAGE = Path("galleries")


def main() -> None:
    """Unified DeepPerson API usage example with User Gallery system."""
    if not SAMPLE_IMAGE.exists():
        print(
            "Sample image missing. Place 'back.jpg' in the repository root to run the demo."
        )
        return

    # Initialize DeepPerson (single entry point)
    dp = DeepPerson()

    # Step 1: Generate multi-modal embeddings from the sample image
    print("=== Step 1: Generate Embeddings ===")
    representation = dp.represent(
        SAMPLE_IMAGE,
        generate_face_embeddings=True,
        return_multi_modal=False,
    )

    subjects = representation["subjects"]
    if not subjects:
        print("No persons detected in the sample image; aborting subsequent steps.")
        return

    primary_subject = subjects[0]
    print(f"Detected {len(subjects)} subject(s)")
    print(
        "Body embedding dim:",
        primary_subject["embedding"].shape,
    )
    face_embedding = primary_subject.get("face_embedding")
    if face_embedding is not None:
        print("Face embedding dim:", face_embedding.shape)
    
    print("representation:", representation)

    # Step 2: Create a user gallery with the detected person
    print("\n=== Step 2: Create User Gallery ===")
    GALLERY_STORAGE.mkdir(parents=True, exist_ok=True)

    try:
        gallery_result = dp.create_gallery(
            user_id="demo_user",
            image_paths=[SAMPLE_IMAGE],
            name="Demo User",
            metadata={"department": "R&D"},
            modality_hints={str(SAMPLE_IMAGE): "BODY"},
            gallery_storage_path=GALLERY_STORAGE,
        )
        print(
            f"Created gallery for user '{gallery_result['user_id']}' "
            f"with {gallery_result['total_images']} image(s)"
        )
    except ValueError as exc:
        print(f"Gallery already exists or error: {exc}")

    # Step 3: Generate embeddings for the user gallery
    print("\n=== Step 3: Generate Gallery Embeddings ===")
    emb_result = dp.represent_gallery(
        user_id="demo_user",
        generate_face_embeddings=True,
        gallery_storage_path=GALLERY_STORAGE,
    )
    print(
        f"Generated {emb_result['generated_embeddings']} embeddings "
        f"({emb_result['face_embeddings_generated']} with face)"
    )

    # Step 4: Check gallery information
    print("\n=== Step 4: Gallery Information ===")
    if dp.gallery_exists("demo_user", gallery_storage_path=GALLERY_STORAGE):
        gallery_info = dp.get_gallery("demo_user", gallery_storage_path=GALLERY_STORAGE)
        print(f"Gallery status: {gallery_info['status']}")
        print(f"Total images: {gallery_info['total_images']}")
        print(f"Modality breakdown: {gallery_info['modality_breakdown']}")

    # Step 5: List all galleries
    print("\n=== Step 5: List All Galleries ===")
    all_galleries = dp.list_galleries(gallery_storage_path=GALLERY_STORAGE)
    print(f"Found {len(all_galleries)} gallery/galleries:")
    for gallery in all_galleries:
        print(f"  - {gallery['user_id']}: {gallery.get('name', 'N/A')}")


if __name__ == "__main__":
    main()
