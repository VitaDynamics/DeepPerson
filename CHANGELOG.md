# Changelog

## [Unreleased]

### Changed
- The `represent` method in `src/api.py` now returns a `RepresentationResult` object instead of a list of `PersonEmbedding` objects. This object contains the `subjects` (list of embeddings), `warnings`, `model_info`, and `face_model_info`.
- The `verify` method in `src/api.py` now returns a `ComparisonResult` object instead of a dictionary.
- Updated `main.py`, `README.md`, and unit tests to reflect the new API return types.
- `DeepPerson` class now supports lazy initialization. `detector` and `embedding_pipeline` are initialized on first use or via `warmup()`.
- Added `warmup()` method to `DeepPerson` to explicitly load models and optionally download/load face models.
- Added `cache_dir` parameter to `DeepPerson` to specify model cache directory.
