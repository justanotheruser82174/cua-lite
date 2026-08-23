#!/bin/bash
# Self-contained render test for start.sh::render_homepage_index.
#
# Run (no backend / docker / flask needed):
#   bash lite/gym/envs/browsergym/scripts/tests/test_render_homepage.sh
#
# Sets fake WA_*/VWA_* env vars with NON-DEFAULT ports (shopping :8615), points
# BROWSERGYM_CACHE at a temp homepage dir holding a copy of the real
# templates/index.html.tmpl, sources start.sh to obtain render_homepage_index,
# runs it, and asserts the rendered index.html has the substituted live URLs (+
# preserved paths) and NO leftover placeholders / default ports. Also exercises
# card-dropping for an UNSET site (map). Prints the rendered links.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
START_SH="$SCRIPT_DIR/start.sh"
REAL_TMPL="$SCRIPT_DIR/../.cache/webarena-homepage/templates/index.html.tmpl"

if [ ! -f "$REAL_TMPL" ]; then
    echo "FAIL: template not found at $REAL_TMPL" >&2
    exit 1
fi

# Temp homepage dir mirroring the served layout.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/webarena-homepage/templates"
cp "$REAL_TMPL" "$TMP/webarena-homepage/templates/index.html.tmpl"

# Fake live URLs with AUTO-PICKED (non-default) ports. MAP intentionally unset
# (OSM not provisioned) → its card must be dropped. CLASSIFIEDS set → VWA card.
export WA_SHOPPING="http://localhost:8615/"
export WA_SHOPPING_ADMIN="http://localhost:8616/admin"
export WA_REDDIT="http://localhost:8617"
export WA_GITLAB="http://localhost:8618"
export WA_WIKIPEDIA="http://localhost:8619/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
unset WA_MAP || true
export VWA_CLASSIFIEDS="http://localhost:8620"

# Source start.sh to pull in render_homepage_index without running services.
# Passing no benchmark arg makes start.sh's dispatch print usage and `return`
# early — but ALL functions are defined by then. BROWSERGYM_CACHE must be set so
# HOMEPAGE_DIR resolves to our temp dir.
export BROWSERGYM_CACHE="$TMP"
# shellcheck disable=SC1090
source "$START_SH" >/dev/null 2>&1 || true

INDEX="$TMP/webarena-homepage/templates/index.html"
render_homepage_index >/dev/null

if [ ! -f "$INDEX" ]; then
    echo "FAIL: render produced no index.html at $INDEX" >&2
    exit 1
fi

fail=0
assert_has() {
    if grep -qF "$1" "$INDEX"; then
        echo "  ok: contains  $1"
    else
        echo "  FAIL: missing  $1" >&2; fail=1
    fi
}
assert_absent() {
    if grep -qF "$1" "$INDEX"; then
        echo "  FAIL: should be absent  $1" >&2; fail=1
    else
        echo "  ok: absent    $1"
    fi
}

echo "── Substituted live URLs (with preserved paths) ──"
assert_has 'href="http://localhost:8615/"'                                                   # shopping
assert_has 'href="http://localhost:8616/admin"'                                              # shopping_admin
assert_has 'href="http://localhost:8617/forums/all"'                                         # reddit
assert_has 'href="http://localhost:8618/explore"'                                            # gitlab
assert_has 'href="http://localhost:8619/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"'  # wikipedia
assert_has 'href="http://localhost:8620/"'                                                   # classifieds

echo "── No leftover placeholders ──"
assert_absent '__SITE_'
assert_absent '<your-server-hostname>'

echo "── No hardcoded DEFAULT ports ──"
for p in 7770 7780 9999 8023 8888 3000 9980; do
    assert_absent ":${p}"
done

echo "── Unprovisioned site (map) card dropped ──"
assert_absent 'OpenStreetMap'
assert_has 'OneStopShop'    # sanity: a provisioned card survived

echo
echo "── Rendered site links ──"
grep -oE 'href="http://localhost:[0-9]+[^"]*"' "$INDEX" || true

echo
if [ "$fail" = "0" ]; then
    echo "PASS: homepage render test"
else
    echo "FAIL: homepage render test" >&2
    exit 1
fi
