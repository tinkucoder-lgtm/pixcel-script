# Git hooks

This directory holds versioned pre-commit hooks for the project.

## Activate (one-time, per clone)

From the repo root:

```bash
git config core.hooksPath .githooks
```

After that, every `git commit` runs the hooks in this folder.

## What's enforced

- **pre-commit**: runs the `font_replacer.py` locked-spec regression test
  (12 assertions, ~0.02s). The file has been silently rewritten twice in
  this project's history; this hook prevents a third occurrence.

## Bypass for one commit

```bash
git commit --no-verify
```

Use sparingly — the hook exists because silent drift in `font_replacer.py`
breaks text rendering in non-obvious ways.

## Requirements

Python 3.x must be available somewhere — the hook tries
`backend/venv/bin/python` first, then `python3`, then `python`. The
regression test uses only the Python stdlib, so no `pip install` needed.
