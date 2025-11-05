"""Core API routes for person recognition.

Implements /represent and /verify endpoints for embedding generation
and identity verification. Supports multipart/form-data and base64 images.

Constitution Principles:
- Pipeline Pattern (II): Detection → Embedding → Verification flow
- Observability (V): Comprehensive logging and error handling
"""

import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile, status
from fastapi.responses import JSONResponse

from fastapi_service.config import get_settings
from fastapi_service.dependencies import DeepPersonDep, RequestIDDep
from fastapi_service.middleware.errors import (
    NoPersonDetectedError,
    ServiceError,
)
from fastapi_service.schemas.represent import (
    Base64RepresentRequest,
    EmbeddingMetadata,
    ModelInfo,
    PersonEmbedding,
    RepresentResponse,
)
from fastapi_service.schemas.verify import (
    Base64VerifyRequest,
    VerifyResponse,
)
from fastapi_service.utils.image_io import (
    decode_base64_image,
    process_upload_file,
    save_image_to_file,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Core API"])

settings = get_settings()


def _save_temp_image(image, suffix: str = ".jpg") -> Path:
    """Save PIL image to temporary file.

    Args:
        image: PIL Image object
        suffix: File suffix (default: .jpg)

    Returns:
        Path to temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix
    )
    temp_path = Path(temp_file.name)
    save_image_to_file(image, temp_path)
    return temp_path


def _convert_person_embedding(
    embedding_dict: dict, source_image: str
) -> PersonEmbedding:
    """Convert DeepPerson embedding dict to Pydantic PersonEmbedding.

    Args:
        embedding_dict: Embedding dictionary from DeepPerson
        source_image: Source image identifier

    Returns:
        PersonEmbedding Pydantic model
    """
    # Extract embedding vector (body embedding)
    embedding_vector = embedding_dict.get("embedding", [])
    if hasattr(embedding_vector, "tolist"):
        embedding_vector = embedding_vector.tolist()

    # Extract face embedding if present
    face_embedding = embedding_dict.get("face_embedding")
    if face_embedding is not None and hasattr(face_embedding, "tolist"):
        face_embedding = face_embedding.tolist()

    # Build metadata
    metadata_dict = embedding_dict.get("metadata", {})

    # Determine modality
    modality = "BODY_FACE" if face_embedding is not None else "BODY_ONLY"

    metadata = EmbeddingMetadata(
        bbox=metadata_dict.get("bbox", [0, 0, 0, 0]),
        confidence=float(metadata_dict.get("confidence", 0.0)),
        hardware=metadata_dict.get("hardware"),
        model_profile_id=metadata_dict.get("model_profile_id"),
        normalization=metadata_dict.get("normalization"),
        timestamp=metadata_dict.get("timestamp", datetime.now()),
        source_image=source_image,
        modality=modality,
        face_confidence=metadata_dict.get("face_confidence"),
        face_bbox=metadata_dict.get("face_bbox"),
    )

    return PersonEmbedding(
        embedding=embedding_vector,
        face_embedding=face_embedding,
        metadata=metadata,
        person_embedding=embedding_dict if metadata_dict.get("return_multi_modal") else None,
    )


@router.post(
    "/represent",
    response_model=RepresentResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate person embeddings",
    description="Extract person embeddings from images with optional face embeddings",
)
async def represent(
    deep_person: DeepPersonDep,
    request_id: RequestIDDep,
    image: Optional[UploadFile] = File(None),
    batch_id: Optional[str] = Form(None),
    detector_backend: Optional[str] = Form(None),
    normalization: str = Form("resnet"),
    batch_size: int = Form(16),
    confidence_threshold: float = Form(0.5),
    generate_face_embeddings: bool = Form(False),
    face_model_name: str = Form("Facenet"),
    face_detector_backend: str = Form("opencv"),
    return_multi_modal: bool = Form(False),
) -> RepresentResponse:
    """Generate person embeddings from image.

    Supports multipart/form-data upload. For base64 images, use POST with JSON body.

    **Request Parameters (Form Data)**:
    - **image**: Image file (required, multipart/form-data)
    - **batch_id**: Optional batch identifier
    - **detector_backend**: Person detection backend
    - **normalization**: Embedding normalization method
    - **batch_size**: Batch size for processing
    - **confidence_threshold**: Minimum detection confidence
    - **generate_face_embeddings**: Enable face embedding generation
    - **face_model_name**: Face recognition model
    - **face_detector_backend**: Face detection backend
    - **return_multi_modal**: Return full PersonEmbedding objects

    **Returns**:
    - List of detected persons with embeddings
    - Model and hardware information
    - Request ID for tracing
    """
    start_time = time.time()

    logger.info(
        f"request_id={request_id} endpoint=/represent "
        f"batch_id={batch_id} generate_face={generate_face_embeddings}"
    )

    if not image:
        raise ServiceError(
            code="MISSING_REQUIRED_FIELD",
            message="Image field is required",
            details={"field": "image"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    try:
        # Process uploaded image
        pil_image = await process_upload_file(image, settings.max_image_size)

        # Save to temporary file for DeepPerson processing
        temp_path = _save_temp_image(pil_image)

        try:
            # Call DeepPerson.represent()
            result = deep_person.represent(
                img_path=str(temp_path),
                detector_backend=detector_backend,
                normalization=normalization,
                batch_size=batch_size,
                confidence_threshold=confidence_threshold,
                generate_face_embeddings=generate_face_embeddings,
                face_model_name=face_model_name,
                face_detector_backend=face_detector_backend,
                return_multi_modal=return_multi_modal,
            )

            # Convert subjects to Pydantic models
            subjects = []
            for emb_dict in result.get("subjects", []):
                person_emb = _convert_person_embedding(
                    emb_dict, source_image=image.filename or "upload"
                )
                subjects.append(person_emb)

            # Check if any persons detected
            if not subjects:
                raise NoPersonDetectedError(confidence_threshold)

            # Build model info
            model_info_dict = result.get("model_info", {})
            model_info = ModelInfo(
                name=model_info_dict.get("name", "resnet50_circle_dg"),
                device=model_info_dict.get("device", "cpu"),
                feature_dim=model_info_dict.get("feature_dim", 2048),
                detector_backend=model_info_dict.get("detector_backend"),
            )

            # Build face model info if present
            face_model_info = None
            if "face_model_info" in result:
                face_info_dict = result["face_model_info"]
                face_model_info = ModelInfo(
                    name=face_info_dict.get("name", face_model_name),
                    device=face_info_dict.get("device", "cpu"),
                    feature_dim=face_info_dict.get("feature_dim", 512),
                )

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                f"request_id={request_id} endpoint=/represent "
                f"status=success subjects_count={len(subjects)} "
                f"duration_ms={duration_ms:.2f}"
            )

            response = RepresentResponse(
                subjects=subjects,
                warnings=result.get("warnings"),
                model_info=model_info,
                face_model_info=face_model_info,
                request_id=request_id,
            )

            return response

        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

    except ServiceError:
        raise
    except Exception as e:
        logger.error(
            f"request_id={request_id} endpoint=/represent error={str(e)}",
            exc_info=True,
        )
        raise ServiceError(
            code="INTERNAL_ERROR",
            message="Failed to process represent request",
            details={"error": str(e)},
        )


@router.post(
    "/represent/base64",
    response_model=RepresentResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate person embeddings (base64)",
    description="Extract person embeddings from base64-encoded images",
)
async def represent_base64(
    deep_person: DeepPersonDep,
    request_id: RequestIDDep,
    request: Base64RepresentRequest,
) -> RepresentResponse:
    """Generate person embeddings from base64-encoded image.

    **Request Body (JSON)**:
    - **base64_image**: Base64-encoded image string (required)
    - All other parameters same as /represent

    **Returns**:
    - Same as /represent endpoint
    """
    start_time = time.time()

    logger.info(
        f"request_id={request_id} endpoint=/represent/base64 "
        f"generate_face={request.generate_face_embeddings}"
    )

    try:
        # Decode base64 image
        pil_image = decode_base64_image(request.base64_image)

        # Save to temporary file
        temp_path = _save_temp_image(pil_image)

        try:
            # Call DeepPerson.represent()
            result = deep_person.represent(
                img_path=str(temp_path),
                detector_backend=request.detector_backend,
                normalization=request.normalization,
                batch_size=request.batch_size,
                confidence_threshold=request.confidence_threshold,
                generate_face_embeddings=request.generate_face_embeddings,
                face_model_name=request.face_model_name,
                face_detector_backend=request.face_detector_backend,
                return_multi_modal=request.return_multi_modal,
            )

            # Convert subjects
            subjects = []
            for emb_dict in result.get("subjects", []):
                person_emb = _convert_person_embedding(emb_dict, "base64_image")
                subjects.append(person_emb)

            if not subjects:
                raise NoPersonDetectedError(request.confidence_threshold)

            # Build model info
            model_info_dict = result.get("model_info", {})
            model_info = ModelInfo(
                name=model_info_dict.get("name", "resnet50_circle_dg"),
                device=model_info_dict.get("device", "cpu"),
                feature_dim=model_info_dict.get("feature_dim", 2048),
                detector_backend=model_info_dict.get("detector_backend"),
            )

            # Build face model info if present
            face_model_info = None
            if "face_model_info" in result:
                face_info_dict = result["face_model_info"]
                face_model_info = ModelInfo(
                    name=face_info_dict.get("name", request.face_model_name),
                    device=face_info_dict.get("device", "cpu"),
                    feature_dim=face_info_dict.get("feature_dim", 512),
                )

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                f"request_id={request_id} endpoint=/represent/base64 "
                f"status=success subjects_count={len(subjects)} "
                f"duration_ms={duration_ms:.2f}"
            )

            return RepresentResponse(
                subjects=subjects,
                warnings=result.get("warnings"),
                model_info=model_info,
                face_model_info=face_model_info,
                request_id=request_id,
            )

        finally:
            if temp_path.exists():
                temp_path.unlink()

    except ServiceError:
        raise
    except Exception as e:
        logger.error(
            f"request_id={request_id} endpoint=/represent/base64 error={str(e)}",
            exc_info=True,
        )
        raise ServiceError(
            code="INTERNAL_ERROR",
            message="Failed to process represent request",
            details={"error": str(e)},
        )


@router.post(
    "/verify",
    response_model=VerifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify person identity",
    description="Compare two images to determine if they show the same person",
)
async def verify(
    deep_person: DeepPersonDep,
    request_id: RequestIDDep,
    img1_path: UploadFile = File(...),
    img2_path: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    detector_backend: Optional[str] = Form(None),
    distance_metric: str = Form("cosine"),
    threshold: Optional[float] = Form(None),
    normalization: str = Form("resnet"),
    enforce_detection: bool = Form(True),
) -> VerifyResponse:
    """Verify if two images show the same person.

    Compares person embeddings from two images using specified distance metric.

    **Request Parameters (Form Data)**:
    - **img1_path**: First image file (required)
    - **img2_path**: Second image file (required)
    - **model_name**: Override model name
    - **detector_backend**: Person detection backend
    - **distance_metric**: Distance metric (cosine, euclidean, euclidean_l2)
    - **threshold**: Verification threshold (None for model default)
    - **normalization**: Embedding normalization method
    - **enforce_detection**: Require person detection

    **Returns**:
    - Verification result (verified boolean)
    - Distance and threshold values
    - Multi-modal fusion scores if face embeddings available
    """
    start_time = time.time()

    logger.info(
        f"request_id={request_id} endpoint=/verify "
        f"metric={distance_metric} threshold={threshold}"
    )

    try:
        # Process both uploaded images
        pil_image1 = await process_upload_file(img1_path, settings.max_image_size)
        pil_image2 = await process_upload_file(img2_path, settings.max_image_size)

        # Save to temporary files
        temp_path1 = _save_temp_image(pil_image1)
        temp_path2 = _save_temp_image(pil_image2)

        try:
            # Call DeepPerson.verify()
            result = deep_person.verify(
                img1_path=str(temp_path1),
                img2_path=str(temp_path2),
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
                threshold=threshold,
                normalization=normalization,
                enforce_detection=enforce_detection,
            )

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                f"request_id={request_id} endpoint=/verify "
                f"status=success verified={result.get('verified')} "
                f"distance={result.get('distance', 0):.4f} "
                f"duration_ms={duration_ms:.2f}"
            )

            # Build response
            response = VerifyResponse(
                verified=result.get("verified", False),
                distance=result.get("distance", 0.0),
                threshold=result.get("threshold", 0.5),
                distance_metric=result.get("distance_metric", distance_metric),
                model=result.get("model", "resnet50_circle_dg"),
                detector_backend=result.get("detector_backend", "yolo"),
                facial_areas=result.get("facial_areas"),
                body_distance=result.get("body_distance", result.get("distance", 0.0)),
                face_distance=result.get("face_distance"),
                fusion_score=result.get("fusion_score"),
                face_weight=result.get("face_weight"),
                body_weight=result.get("body_weight"),
                used_fusion=result.get("used_fusion", False),
                modality_available=result.get(
                    "modality_available", {"body": True, "face": False}
                ),
                warnings=result.get("warnings"),
                request_id=request_id,
            )

            return response

        finally:
            # Clean up temporary files
            if temp_path1.exists():
                temp_path1.unlink()
            if temp_path2.exists():
                temp_path2.unlink()

    except ServiceError:
        raise
    except Exception as e:
        logger.error(
            f"request_id={request_id} endpoint=/verify error={str(e)}",
            exc_info=True,
        )
        raise ServiceError(
            code="INTERNAL_ERROR",
            message="Failed to process verify request",
            details={"error": str(e)},
        )


@router.post(
    "/verify/base64",
    response_model=VerifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify person identity (base64)",
    description="Compare two base64-encoded images to verify identity",
)
async def verify_base64(
    deep_person: DeepPersonDep,
    request_id: RequestIDDep,
    request: Base64VerifyRequest,
) -> VerifyResponse:
    """Verify identity using base64-encoded images.

    **Request Body (JSON)**:
    - **base64_img1**: First image as base64 string (required)
    - **base64_img2**: Second image as base64 string (required)
    - All other parameters same as /verify

    **Returns**:
    - Same as /verify endpoint
    """
    start_time = time.time()

    logger.info(
        f"request_id={request_id} endpoint=/verify/base64 "
        f"metric={request.distance_metric}"
    )

    try:
        # Decode both base64 images
        pil_image1 = decode_base64_image(request.base64_img1)
        pil_image2 = decode_base64_image(request.base64_img2)

        # Save to temporary files
        temp_path1 = _save_temp_image(pil_image1)
        temp_path2 = _save_temp_image(pil_image2)

        try:
            # Call DeepPerson.verify()
            result = deep_person.verify(
                img1_path=str(temp_path1),
                img2_path=str(temp_path2),
                model_name=request.model_name,
                detector_backend=request.detector_backend,
                distance_metric=request.distance_metric,
                threshold=request.threshold,
                normalization=request.normalization,
                enforce_detection=request.enforce_detection,
            )

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                f"request_id={request_id} endpoint=/verify/base64 "
                f"status=success verified={result.get('verified')} "
                f"distance={result.get('distance', 0):.4f} "
                f"duration_ms={duration_ms:.2f}"
            )

            return VerifyResponse(
                verified=result.get("verified", False),
                distance=result.get("distance", 0.0),
                threshold=result.get("threshold", 0.5),
                distance_metric=result.get("distance_metric", request.distance_metric),
                model=result.get("model", "resnet50_circle_dg"),
                detector_backend=result.get("detector_backend", "yolo"),
                facial_areas=result.get("facial_areas"),
                body_distance=result.get("body_distance", result.get("distance", 0.0)),
                face_distance=result.get("face_distance"),
                fusion_score=result.get("fusion_score"),
                face_weight=result.get("face_weight"),
                body_weight=result.get("body_weight"),
                used_fusion=result.get("used_fusion", False),
                modality_available=result.get(
                    "modality_available", {"body": True, "face": False}
                ),
                warnings=result.get("warnings"),
                request_id=request_id,
            )

        finally:
            if temp_path1.exists():
                temp_path1.unlink()
            if temp_path2.exists():
                temp_path2.unlink()

    except ServiceError:
        raise
    except Exception as e:
        logger.error(
            f"request_id={request_id} endpoint=/verify/base64 error={str(e)}",
            exc_info=True,
        )
        raise ServiceError(
            code="INTERNAL_ERROR",
            message="Failed to process verify request",
            details={"error": str(e)},
        )
