---
trigger: always_on
description: Git & Terminal Execution Guidelines for granular commits, PowerShell syntax, and branch safety.
---

# Git & Terminal Execution Guidelines

## 1. Granular Commits Policy
- **Individual Staging & Commits**: When committing multiple changed files individually, stage (`git add <file>`) and commit (`git commit -m "..."`) each file one-by-one with semantic commit prefixes (`feat`, `fix`, `docs`, `style`, `test`, `chore`, `refactor`).
- **Push at the end**: Defer `git push` until all individual file commits are completed in the local working tree.

## 2. Windows PowerShell Syntax
- On Windows PowerShell, avoid using `&&` for command chaining.
- Use `;` or execute commands sequentially as separate invocations.

## 3. Branch Safety & Baseline Preservation Protocol
- **Protected Baseline (`main`)**: Never delete, rewrite history, or merge into `main` without explicit user sign-off.
- **Milestone Tagging**: When branching away or evolving architecture significantly, tag and verify the baseline commit (e.g., `<project>-v1-baseline`) to ensure full recoverability.
- **Pre-Push Inspection**: Before executing `git push`, perform a read-only audit of unpushed commits (`git log origin/<branch>..<branch> --stat`) and confirm with the user.
- **No Blind Pushes**: Never combine branch merges with remote pushes in a single unreviewed step.
