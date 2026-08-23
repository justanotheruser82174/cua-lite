---
name: audit
description: Audit codebase for inconsistencies, redundant code, stale references, and (if staged changes exist) newly introduced bugs.
---

Codebase health audit: find inconsistencies, redundant code, stale references, and (if staged changes exist) newly introduced bugs.

## 1. Consistency

When the same concept is implemented in multiple places, read each and verify alignment:
- Parallel implementations across envs: same data shapes, field names, error handling
- Naming: variable prefixes, dict keys, file names match across code, docs, and comments
- Docstrings, comments, string literals (JSON keys, log messages): describe what the code does now

## 2. Redundant code

Delete: backward-compat shims, dead imports, unused functions, unreachable branches. If it only exists to not break old callers that no longer exist, remove it.

## 3. Staged changes (if `git diff --staged` is non-empty)

Read the full diff and the full file for each change. Check:
- Renames: every reference updated? Grep the old name globally, read each match.
- New code paths: reachable? Edge cases?
- Regex changes: old behavior preserved + new behavior works?
- Dataclass/dict field changes: all constructors, access sites updated?

## Rules

- Report findings as a checklist
- Fix mechanical issues (dead code, stale references, broken renames, obvious bugs) before returning
- For judgement calls (competing conventions, which of two naming patterns to unify to, scope of a refactor), surface the options and ask the user before editing
- Use grep to locate candidates, then **Read each file** to verify — grep alone misses context, docstrings, and partial matches
