# Contributing Guidelines

Welcome to the BINDAS Project!  
To maintain code quality, consistency, and smooth collaboration, please follow the guidelines below when contributing to this repository.

---

## 📌 Branching Strategy

We follow a clean, structured branching model:

- **`main`** → Stable, production-ready code only.
- **`dev`** → Integration branch. All completed features are merged here and tested together before promoting to `main`.
- **`feature/<task-name>`** → Individual branches for developing new features or enhancements.

---

## 📌 Creating a New Feature Branch

1. Ensure you have the latest `dev` branch:
   ```bash
   git checkout dev
   git pull origin dev

2. Create a feature branch:
   ```bash
   git checkout -b feature/<task-name>

3. Push your new branch to the remote:
   ```bash
   git push -u origin feature/<task-name>

Example,
   git checkout -b feature/relationship-engine
   git push -u origin feature/relationship-engine

## 📌 Commit Message Format

Please use clear, descriptive commit messages following this format:

```
<type>: <short description>

[optional longer description if needed]
```

### ✅ Types

- `feat`: New feature  
- `fix`: Bug fix  
- `docs`: Documentation updates  
- `refactor`: Code restructuring without functional change  
- `test`: Adding or updating tests  
- `chore`: Other changes (build setup, dependencies, etc.)

### 💡 Example

```
feat: added relationship detection engine integration
```

---

## 🔀 Merging Workflow

1. **Feature Completion**  
   Merge your `feature/<task-name>` branch into `dev` via a pull request (PR) on GitHub.  
   PR must be reviewed and approved by at least one other team member before merging.

2. **Integration Testing**  
   Once all features for a milestone are merged into `dev`, perform integration testing.

3. **Production Release**  
   After successful testing, merge `dev` into `main`. Tag the release if applicable.

---

## 🔄 Keeping Your Branch Updated

To stay in sync with ongoing development:

```bash
git checkout dev
git pull origin dev
git checkout feature/<task-name>
git merge dev
# Resolve conflicts if any, then:
git push origin feature/<task-name>
```

---

## 🧼 Code Quality Guidelines

- Write meaningful **docstrings** for functions and modules  
- Avoid committing sensitive files like `.env`, virtual environments, or personal config files  
- Use proper **error and exception handling**  

---

✅ Thank you for contributing to **BINDAS Project**!
