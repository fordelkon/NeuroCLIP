---
name: git-commit-guideline
description: Use when planning, reviewing, splitting, squashing, rewriting, or preparing Git commits for a PR, especially when distinguishing local development commits from reviewer-facing PR commits, rebasing onto main, cleaning bug-fix histories, or deciding commit boundaries.
---

# Git Commit Principle

Use this skill to turn messy development history into a clear,
reviewer-friendly commit sequence before opening or updating a PR.

Core principle:

```text
Local commits serve the developer.
PR commits serve reviewers, maintainers, bisect, and future history.
```

Local commits may look like experiment notes. PR commits should read like a
logical patch series: each commit has a cohesive diff, a useful subject, and
enough context for someone else to understand why the change exists.

This skill combines history-shaping practice with commit-message conventions
adapted from Chris Beams' "How to Write a Git Commit Message":
https://chris.beams.io/git-commit

## Operating Modes

### Local Development

Optimize local commits for:

- Safe rollback.
- Small experiments.
- Isolated checkpoints.
- Debugging breadcrumbs.
- Branching from useful intermediate states.
- Preserving enough context to recover from wrong turns.

Acceptable local examples:

```text
debug: log waline stats response
wip: try mounted timing for stats
fix: avoid clearing stats on failed request
style: tighten card spacing
```

These are useful while working, but they usually should not be the final PR
history.

### PR Review

Optimize PR commits for:

- Clear logical progression.
- Independently understandable changes.
- Minimal noise from temporary debugging.
- Reasonable review units.
- Good subject lines and explanatory bodies when needed.
- Bisect-friendly history where each commit ideally builds and passes relevant
  checks.

Reviewer-facing examples:

```text
fix: preserve loaded stats after request failure
fix: handle empty stats payloads
test: cover stats loading failure states
```

## First Pass: Inspect The Branch

Before rewriting, learn the current shape of the branch:

```bash
git status --short --branch
git branch --show-current
git remote -v
git log --oneline --decorate --graph --max-count=30
git diff --stat <base>...HEAD
```

Find the base branch or start commit:

```bash
git merge-base HEAD origin/main
```

If no remote exists, use the local intended base, the initial commit, or the
commit requested by the user.

Do not rewrite shared/public history unless the user explicitly confirms in writing or via a command that history rewriting is acceptable.

## Decide Commit Boundaries

Use these rules when shaping commits:

| Action            | Constraint / Best Practice                                                   |
| ----------------- | ---------------------------------------------------------------------------- |
| **Separate**      | Setup vs. Product Behavior; Refactors vs. Behavior Changes.                  |
| **Separate**      | Tests vs. Unrelated implementation details.                                  |
| **Separate**      | UI, API, data/model, migration, and content (when independently reviewable). |
| **Squash**        | Fixups, debug commits, formatting churn, and tiny corrections.               |
| **Retain Splits** | Only if the split helps review, revert, or bisect.                           |
| **Refactors**     | Put behavior-preserving refactors *before* the behavior change they enable.  |

### Common PR History Shapes

**Feature Work:**

- `chore: initialize project tooling`
- `feat: add content model and page data`
- `test: add site regression coverage`

**Bug Fixes:**

- `test: reproduce stats loading race`
- `fix: preserve existing stats on request failure`

**Refactor + Fix:**

- `refactor: isolate stats request state`
- `fix: preserve previous stats on transient failures`

## Stable Feature Workflow

For well-understood feature work, make small local commits while building:

```text
feat: add comment api type definitions
feat: add comment list component
feat: add comment form component
feat: connect comment form with backend api
fix: handle empty comment list
style: adjust comment card spacing
```

Before PR review, rewrite them into fewer logical commits:

```text
feat: add comment data model and api integration
feat: implement comment list and form UI
fix: handle empty state and polish comment layout
```

Typical cleanup command:

```bash
git fetch origin
git rebase -i origin/main
```

During interactive rebase, use `reword`, `squash`, `fixup`, `edit`, and
`reorder` to transform checkpoints into review steps.

## Bug Fix Workflow

Bug-fix branches are often exploratory. Local history may preserve hypotheses:

```text
debug: log waline stats response
debug: add fallback for empty stats payload
fix: adjust request timing after mounted
debug: test abort controller behavior
fix: avoid overwriting stats during failed request
```

These commits help the developer debug, but they are poor PR commits when they
contain temporary logs, mixed concerns, reverted ideas, or partial fixes.

Before review, reset to the branch start while preserving final file changes:

```bash
git merge-base HEAD origin/main
git reset --soft <branch-start-commit>
git reset
```

Then rebuild logical commits manually:

```bash
git add -p
git commit
git add -p
git commit
git add -p
git commit
```

Use `git add -p` deliberately. If a hunk mixes concerns, split or edit the
hunk instead of accepting it wholesale.

Only use the reset workflow when the branch is private or collaborators have
agreed to rewrite history.

## Rebase Onto Current Main

Before finalizing a PR branch, rebase onto the latest upstream main unless the
project explicitly prefers merge commits:

```bash
git fetch origin
git rebase origin/main
```

This keeps the PR linear and exposes conflicts one commit at a time. Conflict
resolution is easier when each commit has one concern.

If the branch has already been pushed, update it with:

```bash
git push --force-with-lease
```

Prefer `--force-with-lease` over `--force` because it protects remote updates
made by someone else.

## Commit Message Standard

Use the project's existing convention first. If the project has no convention,
use Conventional Commit-style subjects: `type(scope): imperative summary`.

### Subject Line Rules

| Rule                | Description                                | Test / Convention                      |
| ------------------- | ------------------------------------------ | -------------------------------------- |
| **Length**          | Keep near 50 chars, max 72.                | Avoid truncation in tools.             |
| **Mood**            | Use imperative mood.                       | "add", "fix" (not "added" or "fixes"). |
| **Punctuation**     | Do not end with a period.                  | Clean and concise list format.         |
| **Capitalization**  | Follow repo convention.                    | Usually lowercase for `type:`.         |
| **Completion Test** | Read as: "If applied, this commit will..." | "...preserve loaded values".           |

### Examples

| Quality  | Message                                     | Reason                         |
| -------- | ------------------------------------------- | ------------------------------ |
| **Good** | `fix: preserve stats after request failure` | Imperative, clear, concise.    |
| **Good** | `feat: add comment form validation`         | Imperative, clear context.     |
| **Good** | `refactor: isolate Waline stats loader`     | Specific architectural change. |
| **Weak** | `fix stuff`                                 | Too vague.                     |
| **Weak** | `updated components`                        | Past tense, lacks specifics.   |
| **Weak** | `more changes`                              | Provides no value.             |

## When To Add A Body

A one-line commit is fine when the diff is obvious and low risk.

Add a body when the commit changes behavior, fixes a subtle bug, introduces a
tradeoff, removes code, changes data shape, affects compatibility, or would
make a future reader ask "why?"

Format:

```text
subject

Explain the previous behavior and why it was wrong or insufficient.
Explain the new behavior and any important tradeoffs or side effects.

Refs: #123
```

Body rules:

- Put one blank line between subject and body.
- Wrap body text around 72 characters.
- Explain what changed and why; let the diff explain most of how.
- Mention side effects, migrations, compatibility notes, or follow-up risks.
- Put issue IDs, PR links, or trailers at the bottom.

Useful body prompts:

```text
Before this commit...
This failed when...
This commit changes...
The tradeoff is...
This is safe because...
```

## Per-Commit Review Checklist

For each final PR commit, run this mental review:

- Does the subject say what applying the commit does?
- Is the diff one logical change?
- Would this commit be understandable from `git show` alone?
- If reverted alone, would the revert make sense?
- If `git bisect` lands here, is the project in a coherent state?
- Are tests, fixtures, docs, and implementation placed with the right commit?
- Are temporary logs, comments, experiments, and dead code gone?
- Does the body explain non-obvious intent, risk, or tradeoffs?

## Split Or Squash Signals

Split when:

- Refactor and behavior change are mixed.
- Tests cover a separate concern from the implementation.
- UI, API, data, migration, formatting, or content changes are bundled without
  a reason.
- A commit is too large to review without losing the core idea.
- A subject cannot honestly summarize the whole diff.

Squash when:

- A commit only fixes a mistake in an earlier branch commit.
- A style polish only belongs to newly introduced code.
- A debug or exploratory commit has no permanent review value.
- The split creates ceremony without helping review, revert, or bisect.

## Final History Review

Before calling the branch ready:

```bash
git log --oneline --decorate --graph <base>..HEAD
git diff --stat <base>...HEAD
git diff <base>...HEAD
git diff --check <base>...HEAD
```

Check that the history reads in a useful order:

```text
Foundation
Core behavior
Integration
Edge cases and tests
Polish only when it belongs
```

If the final tree must remain identical after rewriting, compare against the
pre-rewrite HEAD:

```bash
git diff --quiet <old-head> HEAD
```

## Guardrails

- Do not rewrite public/shared branch history without coordination.
- Do not hide meaningful review context just to minimize commit count.
- Do not keep temporary debugging commits in a PR unless they intentionally add
  permanent diagnostics.
- Do not combine broad formatting churn with behavioral changes.
- Do not use rebase to paper over unresolved design or test issues.
- Do not claim a commit is independent if it only works because of a later
  commit.
- Do not use `git reset --hard` for cleanup unless the user explicitly requests
  destructive discard.
- Prefer `git push --force-with-lease` when rewritten history must update a
  remote branch.

## Completion Criteria

The work is ready when:

- The branch is based on the intended base branch.
- Commit count matches the complexity of the change.
- Each commit contains cohesive file changes.
- Each subject is concise, imperative, and convention-compliant.
- Commit bodies explain non-obvious why/what context.
- Temporary logs and experimental code are gone.
- Relevant checks have been run.
- The final PR history helps a reviewer understand and maintain the code.
