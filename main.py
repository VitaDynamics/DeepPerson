#!/usr/bin/env python3
"""
DeepPerson Core API - Usage Examples

This script demonstrates the simplified, stateless DeepPerson API which provides:
- Multi-modal person detection and embedding generation
- Identity verification between images with fusion scoring
- Optimized batch processing with 2-5x speedup
- Automatic memory management for large batches

The API is stateless - no gallery management or state tracking required!
"""

from pathlib import Path

from src.api import DeepPerson

SAMPLE_IMAGE = Path("black.jpg")


def main() -> None:
    """Demonstrate the core DeepPerson API functionality."""
    print("=" * 80)
    print("DeepPerson Core API - Demonstration")
    print("=" * 80)
    print("\nThis demo shows the simplified, stateless API with 3 core methods:")
    print("  1. represent()       - Generate multi-modal embeddings from images")
    print("  2. batch_represent() - Optimized batch processing (2-5x faster)")
    print("  3. verify()          - Verify if two images show the same person")
    print()

    # Check for sample image
    if not SAMPLE_IMAGE.exists():
        print(f"⚠️  Sample image missing: {SAMPLE_IMAGE}")
        print("   Please place 'back.jpg' in the repository root to run the demo.")
        print("\n📖 Quick Usage Guide:")
        print("   from src.api import DeepPerson")
        print("   dp = DeepPerson()")
        print("   ")
        print("   # Single image")
        print("   result = dp.represent('image.jpg')")
        print("   ")
        print("   # Batch processing (2-5x faster)")
        print("   batch = dp.batch_represent(['img1.jpg', 'img2.jpg', 'img3.jpg'])")
        print("   ")
        print("   # Identity verification")
        print("   verification = dp.verify('img1.jpg', 'img2.jpg')")
        return

    # Initialize DeepPerson
    print("🔧 Initializing DeepPerson...")
    dp = DeepPerson()
    print(f"   ✓ Model: {dp.model_name}")
    print(f"   ✓ Device: {dp.device}")
    print(f"   ✓ Detector: {dp.detector_backend}")
    print()

    # ============================================================================
    # STEP 1: Generate Multi-Modal Embeddings
    # ============================================================================
    print("=" * 80)
    print("STEP 1: Generate Multi-Modal Embeddings")
    print("=" * 80)
    print("\n📸 Processing image:", SAMPLE_IMAGE)
    print("   Generating body + face embeddings...")
    print()

    representation = dp.represent(
        SAMPLE_IMAGE,
        generate_face_embeddings=True,
        return_multi_modal=False,
    )

    subjects = representation["subjects"]
    if not subjects:
        print("❌ No persons detected in the sample image.")
        return

    primary_subject = subjects[0]
    print(f"✓ Detected {len(subjects)} person(s)")
    print(f"   Body embedding: {primary_subject['embedding'].shape}")

    face_embedding = primary_subject.get("face_embedding")
    if face_embedding is not None:
        print(f"   Face embedding:  {face_embedding.shape} ✓")
    else:
        print("   Face embedding:  Not available")

    print("\n📊 Model Information:")
    print(f"   Model: {representation['model_info']['name']}")
    print(f"   Device: {representation['model_info']['device']}")
    print(f"   Feature Dim: {representation['model_info']['feature_dim']}")

    if representation.get("face_model_info"):
        print(f"\n   Face Model: {representation['face_model_info']['name']}")
        print(
            f"   Face Feature Dim: {representation['face_model_info']['feature_dim']}"
        )

    # Show some metadata
    print("\n📋 Sample Metadata:")
    metadata = primary_subject["metadata"]
    print(f"   Confidence: {metadata['confidence']:.3f}")
    print(f"   Normalization: {metadata['normalization']}")
    print(f"   Modality: {metadata['modality']}")

    # ============================================================================
    # STEP 2: Identity Verification
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Identity Verification")
    print("=" * 80)

    # Self-verification (same image)
    print(f"\n🔍 Verifying image against itself: {SAMPLE_IMAGE.name}")
    verification_result = dp.verify(SAMPLE_IMAGE, SAMPLE_IMAGE)

    print(
        f"\n✓ Result: {'SAME PERSON' if verification_result['verified'] else 'DIFFERENT PERSONS'}"
    )
    print(f"   Distance: {verification_result['distance']:.4f}")
    print(f"   Threshold: {verification_result['threshold']:.4f}")
    print(f"   Metric: {verification_result['distance_metric']}")
    print(f"   Fusion used: {verification_result['used_fusion']}")
    print(f"   Body Distance: {verification_result['body_distance']:.4f}")

    if verification_result.get("face_distance") is not None:
        print(f"   Face Distance: {verification_result['face_distance']:.4f}")

    if verification_result.get("fusion_score") is not None:
        print(f"   Fusion Score: {verification_result['fusion_score']:.4f}")

    print("\n   Modalities available:")
    for modality, available in verification_result["modality_available"].items():
        status = "✓" if available else "✗"
        print(f"     {modality:8s}: {status}")

    if verification_result.get("warnings"):
        print("\n⚠️  Warnings:")
        for warning in verification_result["warnings"]:
            print(f"   - {warning}")

    # ============================================================================
    # STEP 3: Batch Processing with batch_represent()
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Batch Processing with batch_represent()")
    print("=" * 80)

    print("\n📦 Processing multiple images with optimized batch API...")
    # In real usage, these would be different image files
    image_paths = [SAMPLE_IMAGE] * 5
    print(f"   Processing {len(image_paths)} images...")
    print("   Using vectorized operations for 2-5x speedup!")

    batch_result = dp.batch_represent(
        image_paths=image_paths,
        generate_face_embeddings=False,  # Disable for faster demo
        batch_size=8,
    )

    print("\n✓ Batch processing complete!")
    print(
        f"   Images processed: {batch_result['success_count']}/{batch_result['batch_metadata']['total_images']}"
    )
    print(f"   Failed: {batch_result['error_count']}")
    print(f"   Processing time: {batch_result['processing_time']:.3f}s")

    # Show batch metadata
    print("\n📊 Batch Metadata:")
    metadata = batch_result["batch_metadata"]
    print(f"   Batch ID: {metadata['batch_id']}")
    print(f"   Total images: {metadata['total_images']}")
    print(f"   Processed: {metadata['processed_images']}")
    print(f"   Failed: {metadata['failed_images']}")

    # Show timing breakdown
    print("\n⏱️  Timing Breakdown:")
    stages = metadata["processing_stages"]
    print(f"   Detection: {stages['detection_time'] * 1000:.1f}ms")
    print(f"   Body embedding: {stages['body_embedding_time'] * 1000:.1f}ms")

    # Show hardware info
    print("\n💻 Hardware Info:")
    hw = metadata["hardware_info"]
    print(f"   Device: {hw['device_type']}")
    if hw.get("cuda_device"):
        print(f"   GPU: {hw['cuda_device']}")
        print(f"   Memory used: {hw.get('final_memory_gb', 0):.2f}GB")

    # Show per-image results
    print("\n📋 Per-Image Results:")
    for i, result in enumerate(batch_result["results"][:3]):  # Show first 3
        status = "✓" if result["processing_status"] == "success" else "✗"
        num_subjects = len(result.get("subjects", []))
        print(f"   Image {i}: {status} {num_subjects} person(s) detected")
    if len(batch_result["results"]) > 3:
        print(f"   ... and {len(batch_result['results']) - 3} more images")

    # ============================================================================
    # STEP 4: Different Distance Metrics
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Different Distance Metrics")
    print("=" * 80)

    print("\n📏 Testing verification with different distance metrics...")
    print("   (Comparing image to itself, so all should return verified=True)\n")

    metrics = ["cosine", "euclidean", "euclidean_l2"]
    for metric in metrics:
        result = dp.verify(
            SAMPLE_IMAGE,
            SAMPLE_IMAGE,
            distance_metric=metric,
        )
        verified_str = "✓" if result["verified"] else "✗"
        print(
            f"   {metric:15s} | Distance: {result['distance']:6.4f} | "
            f"Threshold: {result['threshold']:5.4f} | {verified_str}"
        )

    # ============================================================================
    # STEP 5: Custom Parameters
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Custom Parameters")
    print("=" * 80)

    print("\n⚙️  Testing with custom verification threshold...")

    # Use a very strict threshold
    strict_result = dp.verify(
        SAMPLE_IMAGE,
        SAMPLE_IMAGE,
        threshold=0.1,  # Very strict
    )
    print("\n   Strict threshold (0.1):")
    print(f"   Distance: {strict_result['distance']:.4f}")
    print(f"   Threshold: {strict_result['threshold']:.4f}")
    print(
        f"   Result: {'✓ Verified' if strict_result['verified'] else '✗ Not verified'}"
    )

    # Use a very lenient threshold
    lenient_result = dp.verify(
        SAMPLE_IMAGE,
        SAMPLE_IMAGE,
        threshold=2.0,  # Very lenient
    )
    print("\n   Lenient threshold (2.0):")
    print(f"   Distance: {lenient_result['distance']:.4f}")
    print(f"   Threshold: {lenient_result['threshold']:.4f}")
    print(
        f"   Result: {'✓ Verified' if lenient_result['verified'] else '✗ Not verified'}"
    )

    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("📚 API Summary")
    print("=" * 80)
    print("\nThe DeepPerson API provides three core stateless methods:\n")
    print("1️⃣  represent() - Generate multi-modal embeddings")
    print("   • Single image or batch processing")
    print("   • Body embeddings (required)")
    print("   • Optional face embeddings (generate_face_embeddings=True)")
    print("   • Configurable normalization and batch size")
    print("   • Returns: subjects with embeddings + metadata\n")
    print("2️⃣  batch_represent() - Optimized batch processing")
    print("   • Process multiple images with 2-5x speedup")
    print("   • Vectorized detection and embedding operations")
    print("   • Automatic memory management and OOM recovery")
    print("   • Smart batch size selection based on GPU memory")
    print("   • Detailed timing and performance metrics")
    print("   • Per-image error handling with partial results")
    print("   • Returns: batch results + metadata + timing breakdown\n")
    print("3️⃣  verify() - Identity verification")
    print("   • Compare two images for same person")
    print("   • Multi-modal fusion scoring (body + face)")
    print("   • Configurable distance metrics: cosine, euclidean, euclidean_l2")
    print("   • Automatic threshold lookup or custom threshold")
    print("   • Returns: verification result + distances + fusion info\n")
    print("✨ Key Features:")
    print("   ✓ Stateless - no gallery management required")
    print("   ✓ Multi-modal - body and face embeddings")
    print("   ✓ Batch processing - handle multiple images efficiently")
    print("   ✓ Multiple metrics - choose the best for your use case")
    print("   ✓ GPU acceleration - automatic CUDA detection")
    print("   ✓ Confidence-based fusion - weighted scoring\n")
    print("🚀 Quick Start:")
    print("   from src.api import DeepPerson")
    print("   dp = DeepPerson()")
    print("   ")
    print("   # Single image")
    print("   result = dp.represent('image.jpg')")
    print("   ")
    print("   # Batch processing (optimized)")
    print("   batch_result = dp.batch_represent(['img1.jpg', 'img2.jpg', 'img3.jpg'])")
    print("   ")
    print("   # Identity verification")
    print("   is_same = dp.verify('img1.jpg', 'img2.jpg')")
    print()


if __name__ == "__main__":
    main()
