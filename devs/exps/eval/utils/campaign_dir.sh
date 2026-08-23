#!/usr/bin/env bash
# Shared campaign-dir resolver for eval run.sh scripts.
#
# Expects callers to define:
#   ENV_ROOT       absolute path to .exps/eval/<env>
#   PIPELINE_PATHS array of paths whose changes alter the campaign pipeline state
# Sets:
#   COMMIT_DIR

resolve_eval_commit_dir() {
  local dir sha changed_paths commit commit_ts allow_eval_commit_dir
  allow_eval_commit_dir="${1:-}"

  COMMIT_DIR=""
  if [ -d "$ENV_ROOT" ]; then
    while IFS= read -r dir; do
      [ -n "$dir" ] || continue
      sha="${dir##*_}"
      # Pipeline diff between $sha and HEAD; empty = pipeline unchanged -> reuse.
      if ! changed_paths="$(git log --format=%h "${sha}..HEAD" -- "${PIPELINE_PATHS[@]}" 2>/dev/null)"; then
        echo "[run.sh] ignoring campaign dir with invalid commit suffix: $dir" >&2
        continue
      fi
      if [ -z "$changed_paths" ]; then
        COMMIT_DIR="$ENV_ROOT/$dir"
        echo "[run.sh] reusing existing campaign dir (pipeline unchanged since $sha): $dir" >&2
        break
      fi
    done < <(
      ls -1 "$ENV_ROOT" 2>/dev/null |
        grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}_[0-9a-f]+$' |
        sort -r
    )
  fi

  if [ -z "$COMMIT_DIR" ]; then
    commit=$(git rev-parse --short HEAD)
    commit_ts=$(git log -1 --date=format:'%Y-%m-%dT%H-%M' --format=%cd HEAD)
    COMMIT_DIR="$ENV_ROOT/${commit_ts}_${commit}"
  fi

  if [ "$allow_eval_commit_dir" = "--allow-eval-commit-dir" ] && [ -n "${EVAL_COMMIT_DIR:-}" ]; then
    if [[ "$EVAL_COMMIT_DIR" = /* ]]; then
      COMMIT_DIR="$EVAL_COMMIT_DIR"
    else
      COMMIT_DIR="$ENV_ROOT/$EVAL_COMMIT_DIR"
    fi
    echo "[run.sh] EVAL_COMMIT_DIR override -> commit_dir=$(basename "$COMMIT_DIR")" >&2
  fi
}
