# Contributing to FreshSense

Thank you for your interest in contributing to FreshSense! This document provides guidelines and instructions for contributing.

## How to Contribute

### 1. Fork the Repository

Click the "Fork" button in the top-right corner of the GitHub repository page to create your own copy.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/FreshSense.git
cd FreshSense
```

### 3. Create a Branch

Create a feature or bugfix branch from `main`:

```bash
git checkout -b feature/my-new-feature
# or
git checkout -b fix/bug-description
```

#### Branch Naming Convention

- **Features**: `feature/descriptive-name`
- **Bug fixes**: `fix/descriptive-name`
- **Documentation**: `docs/descriptive-name`
- **Refactoring**: `refactor/descriptive-name`
- **Performance**: `perf/descriptive-name`

Use lowercase and hyphens. Be descriptive but concise.

### 4. Make Your Changes

#### Coding Style

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write docstrings for all public classes and functions
- Keep functions focused and small
- Use meaningful variable names

#### Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(training): add gradient accumulation support
fix(dataset): handle corrupted images gracefully
docs(readme): update installation instructions
refactor(config): simplify dataclass hierarchy
```

#### Pre-commit Checklist

- [ ] Code follows PEP 8
- [ ] All functions have type hints
- [ ] All public APIs have docstrings
- [ ] No debug code (print statements, pdb)
- [ ] No unused imports or variables
- [ ] Changes tested locally
- [ ] Documentation updated (if applicable)

### 5. Test Your Changes

```bash
# Verify syntax
python -m py_compile src/main.py

# Test imports
python -c "from src.training.trainer import Trainer; print('OK')"

# Run the pipeline (if dataset is available)
python -m src.main
```

### 6. Commit and Push

```bash
git add .
git commit -m "feat(training): add gradient accumulation support"
git push origin feature/my-new-feature
```

### 7. Create a Pull Request

1. Go to your forked repository on GitHub
2. Click "New Pull Request"
3. Select your branch and target `main`
4. Fill out the PR template completely
5. Link any related issues (e.g., "Closes #123")
6. Submit the PR

### 8. Code Review

- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, your PR will be merged

## Pull Request Guidelines

### PR Title

Follow the same convention as commit messages:
```
feat(training): add gradient accumulation support
```

### PR Description

- Clearly describe what the PR does
- Explain why the change is needed
- Link to related issues
- Include before/after comparisons if relevant

### Scope

- Keep PRs focused on a single feature or fix
- Large changes should be broken into smaller, reviewable PRs
- Don't mix refactoring with features

## Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify setup
python -c "import torch, albumentations, cv2, sklearn; print('OK')"
```

## Project Structure

```
FreshSense/
├── configs/              # Configuration files
├── src/                  # Source code
│   ├── main.py          # Entry point
│   ├── models/          # Model definitions
│   ├── training/        # Training loop and utilities
│   ├── preprocessing/   # Data preprocessing
│   ├── inference/       # Inference pipeline
│   └── utils/           # Utilities
├── data/                # Dataset (gitignored)
├── models/              # Model checkpoints (gitignored)
├── logs/                # Log files (gitignored)
├── requirements.txt     # Dependencies
├── README.md           # Project documentation
├── CHANGELOG.md        # Version history
└── ROADMAP.md          # Future plans
```

## Questions?

- Open an issue with the `question` label
- Check existing documentation in `README.md` and `TESTING.md`
- Review closed issues for similar questions

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

Thank you for contributing!