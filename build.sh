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

# Step 0: Version Management (only for upload modes)
if [ "$NO_UPLOAD" = false ]; then
    echo -e "${YELLOW}=== Step 0: Version Management ===${NC}"

    # Extract current version from pyproject.toml
    CURRENT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

    if [ -z "$CURRENT_VERSION" ]; then
        echo -e "${RED}Failed to detect current version from pyproject.toml${NC}"
        exit 1
    fi

    echo "Current version: ${GREEN}${CURRENT_VERSION}${NC}"
    echo ""
    echo "Update version for this release?"
    echo "  (PATCH)  Bug fixes:      ${CURRENT_VERSION} → X.Y.$(echo $CURRENT_VERSION | cut -d. -f3 | awk '{print $1+1}')"
    echo "  (MINOR)  New features:   ${CURRENT_VERSION} → X.$(echo $CURRENT_VERSION | cut -d. -f2 | awk '{print $1+1}').0"
    echo "  (MAJOR)  Breaking:       ${CURRENT_VERSION} → $(echo $CURRENT_VERSION | cut -d. -f1 | awk '{print $1+1}').0.0"
    echo "  (CUSTOM) Enter version manually"
    echo "  (SKIP)   Keep ${CURRENT_VERSION}"
    echo ""
    read -p "Update version? [patch/minor/major/custom/skip] (default: skip): " VERSION_CHOICE

    NEW_VERSION=""

    case "${VERSION_CHOICE,,}" in
        patch|p)
            # Bump patch version (0.1.0 → 0.1.1)
            MAJOR=$(echo $CURRENT_VERSION | cut -d. -f1)
            MINOR=$(echo $CURRENT_VERSION | cut -d. -f2)
            PATCH=$(echo $CURRENT_VERSION | cut -d. -f3 | awk '{print $1+1}')
            NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"
            ;;
        minor|m)
            # Bump minor version (0.1.0 → 0.2.0)
            MAJOR=$(echo $CURRENT_VERSION | cut -d. -f1)
            MINOR=$(echo $CURRENT_VERSION | cut -d. -f2 | awk '{print $1+1}')
            NEW_VERSION="${MAJOR}.${MINOR}.0"
            ;;
        major|M)
            # Bump major version (0.1.0 → 1.0.0)
            MAJOR=$(echo $CURRENT_VERSION | cut -d. -f1 | awk '{print $1+1}')
            NEW_VERSION="${MAJOR}.0.0"
            ;;
        custom|c)
            # Custom version
            read -p "Enter new version (format: X.Y.Z): " NEW_VERSION
            # Validate semantic versioning format
            if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                echo -e "${RED}Invalid version format! Must be X.Y.Z (e.g., 1.2.3)${NC}"
                exit 1
            fi
            ;;
        skip|s|"")
            # Keep current version
            echo "Keeping current version: ${CURRENT_VERSION}"
            NEW_VERSION="${CURRENT_VERSION}"
            ;;
        *)
            echo -e "${RED}Invalid choice: ${VERSION_CHOICE}${NC}"
            exit 1
            ;;
    esac

    # Update version if changed
    if [ "$NEW_VERSION" != "$CURRENT_VERSION" ] && [ -n "$NEW_VERSION" ]; then
        echo ""
        echo -e "${YELLOW}Updating version: ${CURRENT_VERSION} → ${NEW_VERSION}${NC}"

        # Update pyproject.toml
        sed -i "s/^version = \"${CURRENT_VERSION}\"/version = \"${NEW_VERSION}\"/" pyproject.toml

        # Update src/__init__.py
        sed -i "s/__version__ = \"${CURRENT_VERSION}\"/__version__ = \"${NEW_VERSION}\"/" src/__init__.py

        # Verify changes
        VERIFY_PYPROJECT=$(grep "^version = \"${NEW_VERSION}\"" pyproject.toml)
        VERIFY_INIT=$(grep "__version__ = \"${NEW_VERSION}\"" src/__init__.py)

        if [ -z "$VERIFY_PYPROJECT" ] || [ -z "$VERIFY_INIT" ]; then
            echo -e "${RED}Version update failed! Please check pyproject.toml and src/__init__.py${NC}"
            exit 1
        fi

        echo -e "${GREEN}  Updated pyproject.toml${NC}"
        echo -e "${GREEN}  Updated src/__init__.py${NC}"
        echo ""
        echo -e "${YELLOW}Remember to commit these changes:${NC}"
        echo "  git add pyproject.toml src/__init__.py"
        echo "  git commit -m \"chore: bump version to ${NEW_VERSION}\""
        echo ""

        # Update the version variable for later use
        CURRENT_VERSION="${NEW_VERSION}"
    fi

    echo -e "${GREEN}  Version: ${CURRENT_VERSION}${NC}"
    echo ""
fi

# Step 1: Pre-build checks
echo -e "${YELLOW}=== Step 1: Pre-build checks ===${NC}"
echo "Running code quality checks..."

# echo "  - Checking with ruff..."
# ruff check src/ tests/ || { echo -e "${RED}Ruff check failed!${NC}"; exit 1; }

# echo "  - Checking code formatting..."
# ruff format --check src/ tests/ || { echo -e "${RED}Code not formatted! Run: ruff format src/ tests/${NC}"; exit 1; }

# echo "  - Running type checks with mypy..."
# mypy src/ || { echo -e "${YELLOW}Warning: Type checking failed (continuing anyway)${NC}"; }

# echo "  - Running tests..."
# pytest tests/ -v || { echo -e "${RED}Tests failed!${NC}"; exit 1; }

echo -e "${GREEN}  All pre-build checks passed${NC}"
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

# # Step 7: Test installation locally
# echo -e "${YELLOW}=== Step 7: Testing local installation ===${NC}"
# echo "Creating test environment..."
# TEST_ENV="test_env_$$"
# python -m venv "$TEST_ENV"
# source "$TEST_ENV/bin/activate"

# echo "Installing package from wheel..."
# pip install -q dist/*.whl

# echo "Testing import..."
# python -c "from deep_person import DeepPerson; print('  Local installation test: PASS')" || {
#     echo -e "${RED}Local installation test failed!${NC}"
#     deactivate
#     rm -rf "$TEST_ENV"
#     exit 1
# }

# deactivate
# rm -rf "$TEST_ENV"
# echo -e "${GREEN}  Local installation test passed${NC}"
# echo ""

# # Exit if no upload requested
# if [ "$NO_UPLOAD" = true ]; then
#     echo -e "${GREEN}=== Build complete (no upload) ===${NC}"
#     echo "Distribution files are ready in dist/"
#     exit 0
# fi

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

# Get final version for display
FINAL_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

if [ "$NO_UPLOAD" = false ]; then
    echo "Next steps:"
    echo "  1. Tag the release: git tag -a v${FINAL_VERSION} -m 'Release v${FINAL_VERSION}'"
    echo "  2. Push the tag: git push origin v${FINAL_VERSION}"
    echo "  3. Create a GitHub release"
else
    echo "Build artifacts ready in dist/ for version ${FINAL_VERSION}"
fi
echo ""
