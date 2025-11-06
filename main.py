from pathlib import Path

from src.api import DeepPerson


SAMPLE_IMAGE = Path("back.jpg")


def main() -> None:
    """DeepPerson Core API Usage Examples.

    Demonstrates the stateless API for:
    - Multi-modal person detection and embedding generation
    - Identity verification between images
    - Batch processing capabilities
    """
    if not SAMPLE_IMAGE.exists():
        print(
            "Sample image missing. Place 'back.jpg' in the repository root to run the demo."
        )
        return

    # Initialize DeepPerson
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

    print("Model info:", representation["model_info"])
    if representation.get("face_model_info"):
        print("Face model info:", representation["face_model_info"])

    # Step 2: Verify images (self-verification)
    print("\n=== Step 2: Verify Images (Self-Verification) ===")
    verification_result = dp.verify(SAMPLE_IMAGE, SAMPLE_IMAGE)
    print(f"Self-verification result: {verification_result['verified']}")
    print(f"Distance: {verification_result['distance']:.4f}")
    print(f"Threshold: {verification_result['threshold']:.4f}")
    print(f"Model used: {verification_result['model']}")
    print(f"Fusion used: {verification_result.get('used_fusion', False)}")
    print(f"Modality available: {verification_result['modality_available']}")
    if verification_result.get('fusion_score') is not None:
        print(f"Fusion score: {verification_result['fusion_score']:.4f}")
    if verification_result.get('warnings'):
        print(f"Warnings: {verification_result['warnings']}")

    # Step 3: Batch processing example
    print("\n=== Step 3: Batch Processing ===")
    # Create dummy image paths (in real usage, these would be actual image files)
    image_paths = [SAMPLE_IMAGE] * 3  # Using same image for demo
    batch_result = dp.represent(
        image_paths,
        generate_face_embeddings=False,  # Disable for faster demo
        batch_size=4,
    )
    print(f"Batch processed {len(image_paths)} image(s)")
    print(f"Total subjects detected: {len(batch_result['subjects'])}")
    print("Model info:", batch_result["model_info"])

    # Step 4: Different distance metrics
    print("\n=== Step 4: Verification with Different Metrics ===")
    metrics = ["cosine", "euclidean", "euclidean_l2"]
    for metric in metrics:
        result = dp.verify(
            SAMPLE_IMAGE,
            SAMPLE_IMAGE,
            distance_metric=metric,
        )
        print(f"{metric:15s} - Distance: {result['distance']:.4f}, "
              f"Verified: {result['verified']}")

    print("\n=== Core API Summary ===")
    print("✓ represent(): Generate multi-modal embeddings (body + optional face)")
    print("✓ verify(): Identity verification with fusion scoring")
    print("✓ Batch processing support")
    print("✓ Multiple distance metrics (cosine, euclidean, euclidean_l2)")
    print("\nAll operations are stateless - no gallery management required!")


if __name__ == "__main__":
    main()
