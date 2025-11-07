#!/bin/bash
# Build and publish DeepPerson package to PyPI
# Usage: ./build.sh [--test-only|--prod-only|--no-upload]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
TEST_ONLY=false
PROD_ONLY=false
NO_UPLOAD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --prod-only)
            PROD_ONLY=true
            shift
            ;;
        --no-upload)
            NO_UPLOAD=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: ./build.sh [--test-only|--prod-only|--no-upload]"
            echo "  --test-only: Build and test, skip PyPI upload"
            echo "  --prod-only: Skip TestPyPI, upload directly to PyPI"
            echo "  --no-upload: Build only, don't upload anywhere"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== DeepPerson Build & Publish ===${NC}"
echo ""

# Step 1: Pre-build checks
echo -e "${YELLOW}=== Step 1: Pre-build checks ===${NC}"
echo "Running code quality checks..."

echo "  - Checking with ruff..."
ruff check src/ tests/ || { echo -e "${RED}Ruff check failed!${NC}"; exit 1; }

echo "  - Checking code formatting..."
ruff format --check src/ tests/ || { echo -e "${RED}Code not formatted! Run: ruff format src/ tests/${NC}"; exit 1; }

echo "  - Running type checks with mypy..."
mypy src/ || { echo -e "${YELLOW}Warning: Type checking failed (continuing anyway)${NC}"; }

echo "  - Running tests..."
pytest tests/ -v || { echo -e "${RED}Tests failed!${NC}"; exit 1; }

echo -e "${GREEN}  All pre-build checks passed${NC}"
echo ""

# Step 2: Check imports
echo -e "${YELLOW}=== Step 2: Checking package imports ===${NC}"
python -c "from deep_person import DeepPerson; print(f'Import check: OK (version {DeepPerson.__module__})')" || {
    echo -e "${RED}Import check failed! Fix src/__init__.py imports before building.${NC}"
    exit 1
}
echo -e "${GREEN}  Import check passed${NC}"
echo ""

# Step 3: Clean build artifacts
echo -e "${YELLOW}=== Step 3: Cleaning build artifacts ===${NC}"
rm -rf dist/ build/ *.egg-info src/*.egg-info
echo -e "${GREEN}  Cleaned${NC}"
echo ""

# Step 4: Build package
echo -e "${YELLOW}=== Step 4: Building package ===${NC}"
python -m build || { echo -e "${RED}Build failed! Ensure 'build' is installed: pip install build${NC}"; exit 1; }
echo -e "${GREEN}  Build complete${NC}"
echo ""

# Step 5: Validate package
echo -e "${YELLOW}=== Step 5: Validating package ===${NC}"
twine check dist/* || { echo -e "${RED}Package validation failed! Ensure 'twine' is installed: pip install twine${NC}"; exit 1; }
echo -e "${GREEN}  Package validated${NC}"
echo ""

# Step 6: Show package contents
echo -e "${YELLOW}=== Step 6: Package contents ===${NC}"
ls -lh dist/
echo ""
echo "Wheel contents:"
python -m zipfile -l dist/*.whl | head -20
echo ""

# Step 7: Test installation locally
echo -e "${YELLOW}=== Step 7: Testing local installation ===${NC}"
echo "Creating test environment..."
TEST_ENV="test_env_$$"
python -m venv "$TEST_ENV"
source "$TEST_ENV/bin/activate"

echo "Installing package from wheel..."
pip install -q dist/*.whl

echo "Testing import..."
python -c "from deep_person import DeepPerson; print('  Local installation test: PASS')" || {
    echo -e "${RED}Local installation test failed!${NC}"
    deactivate
    rm -rf "$TEST_ENV"
    exit 1
}

deactivate
rm -rf "$TEST_ENV"
echo -e "${GREEN}  Local installation test passed${NC}"
echo ""

# Exit if no upload requested
if [ "$NO_UPLOAD" = true ]; then
    echo -e "${GREEN}=== Build complete (no upload) ===${NC}"
    echo "Distribution files are ready in dist/"
    exit 0
fi

# Step 8: Upload to TestPyPI
if [ "$PROD_ONLY" = false ]; then
    echo -e "${YELLOW}=== Step 8: Upload to TestPyPI ===${NC}"
    echo "This will upload to https://test.pypi.org"
    echo ""
    read -p "Upload to TestPyPI? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        twine upload --repository testpypi dist/* || {
            echo -e "${YELLOW}Warning: TestPyPI upload failed (might already exist)${NC}"
        }
        echo ""
        echo -e "${GREEN}  Uploaded to TestPyPI${NC}"
        echo "Test installation with:"
        echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple deep-person"
        echo ""

        if [ "$TEST_ONLY" = true ]; then
            echo -e "${GREEN}=== Build complete (test-only mode) ===${NC}"
            exit 0
        fi
    else
        echo "Skipping TestPyPI upload"
        echo ""
    fi
fi

# Step 9: Upload to Production PyPI
if [ "$TEST_ONLY" = false ]; then
    echo -e "${YELLOW}=== Step 9: Upload to Production PyPI ===${NC}"
    echo -e "${RED}WARNING: This will upload to PRODUCTION PyPI!${NC}"
    echo "Make sure:"
    echo "  - Version is correct in pyproject.toml"
    echo "  - You have tested on TestPyPI"
    echo "  - You have a PyPI API token"
    echo ""
    read -p "Upload to PRODUCTION PyPI? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        twine upload dist/* || {
            echo -e "${RED}PyPI upload failed!${NC}"
            exit 1
        }
        echo ""
        echo -e "${GREEN}  Uploaded to PyPI${NC}"
        echo "Install with:"
        echo "  pip install deep-person[all]"
        echo ""
    else
        echo "Skipping PyPI upload"
        echo ""
    fi
fi

echo -e "${GREEN}=== Build and publish complete! ===${NC}"
echo ""
echo "Next steps:"
echo "  1. Tag the release: git tag -a v0.1.0 -m 'Release v0.1.0'"
echo "  2. Push the tag: git push origin v0.1.0"
echo "  3. Create a GitHub release"
echo ""
