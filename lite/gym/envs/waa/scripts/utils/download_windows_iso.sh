#!/usr/bin/env bash
# Download the pinned official Windows 11 Enterprise Evaluation 25H2 EN-US x64
# ISO and verify its SHA256. This is Microsoft's freely downloadable, time-limited
# evaluation software; using it is subject to Microsoft's evaluation license
# (https://www.microsoft.com/en-us/evalcenter/download-windows-11-enterprise).
#
# Run: uv run --no-sync bash lite/gym/envs/waa/scripts/utils/download_windows_iso.sh [output.iso]
set -euo pipefail

OUTPUT="${1:-$HOME/.cache/cua-lite/waa/iso/windows11-enterprise-eval-en-us-x64.iso}"
ISO_URL="https://software-static.download.prss.microsoft.com/dbazure/888969d5-f34g-4e03-ac9d-1f9786c66749/26200.6584.250915-1905.25h2_ge_release_svc_refresh_CLIENTENTERPRISEEVAL_OEMRET_x64FRE_en-us.iso"
EXPECTED_SHA="a61adeab895ef5a4db436e0a7011c92a2ff17bb0357f58b13bbc4062e535e7b9"

OUTPUT="$(realpath -m "$OUTPUT")"
PART="${OUTPUT}.part"
command -v curl >/dev/null || { echo "ISO download requires curl." >&2; exit 2; }
command -v sha256sum >/dev/null || { echo "ISO download requires sha256sum." >&2; exit 2; }

verify_iso() {
  [[ -s "$1" ]] || return 1
  [[ "$(sha256sum "$1" | awk '{print tolower($1)}')" == "$EXPECTED_SHA" ]]
}

if verify_iso "$OUTPUT"; then
  echo "Pinned official Windows ISO already present: $OUTPUT"
  printf '%s\n' "$OUTPUT"
  exit 0
fi
if [[ -e "$OUTPUT" ]]; then
  echo "Existing ISO does not match the pinned Windows 11 25H2 SHA256: $OUTPUT" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
curl --fail --location --retry 10 --retry-all-errors --continue-at - --output "$PART" "$ISO_URL"
if ! verify_iso "$PART"; then
  actual="$(sha256sum "$PART" | awk '{print tolower($1)}')"
  echo "Windows ISO SHA256 mismatch: expected $EXPECTED_SHA, got $actual" >&2
  exit 1
fi
mv -f "$PART" "$OUTPUT"
echo "Downloaded pinned official Windows ISO: $OUTPUT"
printf '%s\n' "$OUTPUT"
