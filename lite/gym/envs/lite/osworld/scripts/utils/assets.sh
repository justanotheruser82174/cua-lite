#!/usr/bin/env bash
# Manage lite.osworld runtime asset bundles.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OSWORLD_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CACHE_ROOT="$OSWORLD_DIR/.cache/assets"
PULLED_ROOT="$CACHE_ROOT/pulled"
BUILD_ROOT="$CACHE_ROOT/build"
TMP_ROOT="$CACHE_ROOT/.tmp"

component="${2:-synth}"

log() { echo "[lite.osworld assets] $*" >&2; }
die() { echo "[lite.osworld assets] ERROR: $*" >&2; exit 1; }

identity() {
    python "$SCRIPT_DIR/asset_lock.py" identity "$OSWORLD_DIR" "$component"
}

repo_revision_path() {
    python "$SCRIPT_DIR/asset_lock.py" fields "$OSWORLD_DIR" "$component"
}

component_dir() {
    echo "$PULLED_ROOT/$component"
}

check_component() {
    local dir expected
    dir="$(component_dir)"
    expected="$(identity)"
    [ -d "$dir" ] || die "$component asset missing: $dir"
    [ -f "$dir/.complete" ] || die "$component asset incomplete: $dir/.complete missing"
    [ -f "$dir/.asset_identity" ] || die "$component asset missing identity: $dir/.asset_identity"
    [ "$(cat "$dir/.asset_identity")" = "$expected" ] || die "$component asset identity is stale"
    [ -f "$dir/MANIFEST.csv" ] || die "$component asset MANIFEST.csv missing"
    log "$component assets fresh ($dir)"
}

pull_component() {
    local repo rev path expected dir tmp
    mapfile -t pin < <(repo_revision_path)
    repo="${pin[0]}"
    rev="${pin[1]}"
    path="${pin[2]}"
    expected="$(identity)"
    dir="$(component_dir)"

    if [ -f "$dir/.complete" ] && [ -f "$dir/.asset_identity" ] \
        && [ "$(cat "$dir/.asset_identity")" = "$expected" ]; then
        log "$component assets up to date (${rev:0:12}); skipping download."
        return 0
    fi

    tmp="$TMP_ROOT/pull-$component-$$"
    rm -rf "$tmp"
    mkdir -p "$tmp"
    log "pulling $component from HF $repo@$rev path=$path"
    python - "$repo" "$rev" "$path" "$tmp" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, rev, path, out = sys.argv[1:5]
allow_patterns = None if path == "." else [f"{path}/**", path]
snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    revision=rev,
    allow_patterns=allow_patterns,
    local_dir=out,
)
PY
    if [ "$path" != "." ]; then
        [ -d "$tmp/$path" ] || die "downloaded snapshot missing component path: $path"
        mkdir -p "$tmp.__component"
        cp -a "$tmp/$path"/. "$tmp.__component"/
        rm -rf "$tmp"
        mv "$tmp.__component" "$tmp"
    fi
    [ -f "$tmp/MANIFEST.csv" ] || die "downloaded $component asset has no MANIFEST.csv"
    printf '%s\n' "$expected" > "$tmp/.asset_identity"
    touch "$tmp/.complete"
    mkdir -p "$PULLED_ROOT"
    rm -rf "$dir.__previous__"
    if [ -e "$dir" ]; then
        mv "$dir" "$dir.__previous__"
    fi
    mv "$tmp" "$dir"
    rm -rf "$dir.__previous__"
    log "$component assets ready in $dir"
}

build_component() {
    local manifest out ok skip fail path license url usage size
    manifest="$OSWORLD_DIR/data/assets/synth/MANIFEST.csv"
    out="$BUILD_ROOT/$component"
    [ "$component" = "synth" ] || die "build is only implemented for synth assets"
    [ -f "$manifest" ] || die "missing manifest: $manifest"
    rm -rf "$out"
    mkdir -p "$out"
    cp "$manifest" "$out/MANIFEST.csv"
    ok=0
    skip=0
    fail=0
    while IFS=, read -r path license url usage; do
        [[ "$path" == "asset_path" ]] && continue
        [[ "$path" == \#* ]] && continue
        [[ -z "$path" ]] && continue
        if [[ -f "$out/$path" ]]; then
            skip=$((skip + 1))
            continue
        fi
        mkdir -p "$(dirname "$out/$path")"
        if curl -sLfo "$out/$path" --max-time 60 "$url"; then
            size=$(stat -c%s "$out/$path" 2>/dev/null || stat -f%z "$out/$path")
            if [[ "$size" -gt 500 ]]; then
                ok=$((ok + 1))
            else
                rm -f "$out/$path"
                echo "too small: $path ($size B)" >&2
                fail=$((fail + 1))
            fi
        else
            echo "curl failed: $path" >&2
            fail=$((fail + 1))
        fi
    done < "$manifest"
    log "build complete: $ok fetched, $skip already present, $fail failed -> $out"
    [ "$fail" -eq 0 ] || exit 1
    printf '%s\n' "$(identity)" > "$out/.build_identity"
    touch "$out/.complete"
}

push_component() {
    local repo rev out expected
    mapfile -t pin < <(repo_revision_path)
    repo="${pin[0]}"
    rev="${pin[1]}"
    expected="$(identity)"
    out="$BUILD_ROOT/$component"
    [ -d "$out" ] || die "missing build candidate: $out"
    [ -f "$out/.complete" ] || die "build candidate incomplete: $out"
    [ -f "$out/.build_identity" ] || die "build candidate missing identity"
    [ "$(cat "$out/.build_identity")" = "$expected" ] || die "build identity does not match asset lock"
    log "uploading $component build candidate to HF dataset $repo"
    python - "$repo" "$out" <<'PY'
import sys
from huggingface_hub import upload_folder

repo, folder = sys.argv[1:3]
upload_folder(repo_id=repo, repo_type="dataset", folder_path=folder, path_in_repo=".")
PY
    log "upload complete; update assets.lock.yaml revision from HF commit, expected currently $rev"
}

status_component() {
    local dir expected state
    dir="$(component_dir)"
    expected="$(identity)"
    state="MISSING"
    if [ -d "$dir" ]; then
        state="STALE"
        if [ -f "$dir/.complete" ] && [ -f "$dir/.asset_identity" ] \
            && [ "$(cat "$dir/.asset_identity")" = "$expected" ]; then
            state="FRESH"
        fi
    fi
    echo "$component asset : $state ($dir)"
}

case "${1:-pull}" in
    pull) pull_component ;;
    check) check_component ;;
    status) status_component ;;
    build) build_component ;;
    push) push_component ;;
    *)
        echo "Usage: $0 [pull|check|status|build|push] [component]" >&2
        exit 1
        ;;
esac
