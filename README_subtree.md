# DeepPerson Subtree Integration Guide

This directory contains the DeepPerson component integrated as a **git subtree** from the external repository [VitaDynamics/DeepPerson](https://github.com/VitaDynamics/DeepPerson).

## What is a Git Subtree?

A git subtree allows you to embed a separate Git repository as a subdirectory of your main repository while maintaining its own commit history. Changes can be pushed and pulled between the subtree and its source repository independently.

## Working with the Subtree

### Pushing Changes to Source Repository

If you've made changes to the component in this subtree and want to push them back to the source repository:

```bash
# From the main Vbot repository root
git subtree push --prefix=components/deep_person deep_person main
```

**What this does:**
- `--prefix=components/deep_person`: Specifies which directory contains the subtree
- `deep_person`: The remote name for the source repository
- `main`: The branch to push to in the source repository

**Example workflow:**
```bash
# 1. Make changes to files in components/deep_person/
cd /path/to/Vbot

# 2. Commit changes to the main repository
git add components/deep_person/
git commit -m "Update deep_person component"

# 3. Push changes to source repository
git subtree push --prefix=components/deep_person deep_person main

# 4. The output will show commits being pushed to the source repo
```

**Troubleshooting push issues:**
```bash
# If you get "no new commits" error, force push
git subtree push --prefix=components/deep_person deep_person main --force

# To push to a different branch
git subtree push --prefix=components/deep_person deep_person develop
```

### Pulling Updates from Source Repository

To pull the latest updates from the source repository into your subtree:

```bash
# From the main Vbot repository root
git subtree pull --prefix=components/deep_person deep_person main --squash
```

**What this does:**
- `--prefix=components/deep_person`: Specifies which directory contains the subtree
- `deep_person`: The remote name for the source repository
- `main`: The branch to pull from in the source repository
- `--squash`: Merges the changes into a single commit (recommended for subtrees)

**Example workflow:**
```bash
# 1. Fetch the latest from source repository
git fetch deep_person main

# 2. Pull changes into the subtree
git subtree pull --prefix=components/deep_person deep_person main --squash

# 3. This creates a merge commit in your main repository

# 4. If there are conflicts, resolve them and commit
git add components/deep_person/
git commit -m "Resolve conflicts in deep_person merge"
```

**Alternative pull without squashing:**
```bash
# Pull with full commit history (more complex history)
git subtree pull --prefix=components/deep_person deep_person main
```

**Pull from a different branch:**
```bash
# Pull from development branch
git subtree pull --prefix=components/deep_person deep_person develop --squash
```

### Verifying Subtree Status

Check if your local subtree is up-to-date with the source:

```bash
# Fetch the latest from source repository
git fetch deep_person main

# Compare your subtree with the remote
git diff --subtree=components/deep_person deep_person/main
```

## Subtree Setup

If you need to re-add this component as a subtree (for example, in a new clone):

```bash
# Add the remote (if not already present)
git remote add deep_person git@github.com:VitaDynamics/DeepPerson.git

# Create the subtree
git subtree add --prefix=components/deep_person deep_person main --squash
```

## Component Usage

This component can be imported and used in your Vbot application:

```python
from components.deep_person.api import DeepPerson

# Initialize the component
dp = DeepPerson()

# Use for person detection and re-identification
result = dp.represent("path/to/person_image.jpg")
```

See the main [README.md](./README.md) for detailed API documentation.

## Important Notes

1. **Commit History**: The subtree maintains its own commit history separate from the main repository
2. **Conflicts**: When pulling updates, you may encounter conflicts. Resolve them as you would with any merge conflict
3. **Large Changes**: For major updates, it's recommended to use the subtree pull without `--squash` to preserve commit history
4. **Remote Name**: The remote is named `deep_person`. You can verify this with: `git remote -v`

## Troubleshooting

### Remote Not Found
If you see "remote 'deep_person' not found":
```bash
git remote add deep_person git@github.com:VitaDynamics/DeepPerson.git
```

### Permission Denied
If you get authentication errors when pushing/pulling:
- Ensure you have write access to the VitaDynamics/DeepPerson repository
- Use SSH keys or token-based authentication with GitHub

### Out of Sync
If your subtree is significantly out of sync:
```bash
# Force pull (use with caution - may overwrite local changes)
git subtree pull --prefix=components/deep_person deep_person main --squash --strategy-option=theirs
```

## References

- [Git Subtree Documentation](https://git-scm.com/book/en/v2/Git-Tools-Advanced-Merging#_subtree_merge)
- [Source Repository](https://github.com/VitaDynamics/DeepPerson)
- Component Main Docs: [README.md](./README.md)
