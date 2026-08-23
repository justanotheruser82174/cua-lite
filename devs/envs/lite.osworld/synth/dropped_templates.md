# Dropped synth templates — the 2026-05-12 validation record

The **decisions** live in code: `_DROPPED_TEMPLATE_IDS` in
[/lite/gym/envs/lite/osworld/src/gen/train/synth/catalog.py](/lite/gym/envs/lite/osworld/src/gen/train/synth/catalog.py)
is the single owner of which `(domain, template_id)` pairs are excluded, and
`_apply_drop_filter` is the only reader. This file holds the **evidence** for those
decisions, which was previously pasted into that data file as 91 lines of comments.

It is kept because it is the only surviving record: `d_imp_01_0001`, the "69-FN
cascade-clear" estimate and the `xcu-seed` reasoning appear nowhere else in the repo, so
deleting the block outright would have destroyed the rationale while leaving the drops
in place. Verbatim below; see [/devs/envs/lite.osworld/synth/AGENTS.md](/devs/envs/lite.osworld/synth/AGENTS.md)
for the V1–V5 pipeline these verdicts came from.

---

=====================================================================
validation (2026-05-12) — replay+screenshot
verified additions. All entries here have hard evidence (predicate
match-count / pHash distance / sha256 / screenshot).
=====================================================================
[verified and removed 2026-05-12] 6 calc bugs:
- B2 f_calc_30 / f_calc_37 (CSV country-filter): libreoffice_calc.py:1244-1278 +
:1432-1474 — added `_WB_AGG` set (47 codes) + 4 income-group labels filter,
sort by ranking column desc, top-30. First 5 = USA/CHN/JPN/DEU/IND.
- B4 f_calc_91 / f_calc_100 (chart-eval gap): libreoffice_calc.py:4055-4090 +
:7585-7625 — gold emits real BarChart; `{"type":"chart",
"chart_props":["type"]}` rule added. Smoke: no-op → 0.0, correct → 1.0.
- B8/B9 f_calc_29 vacuous predicate: :5901-5910 — P[0] `>8.0` → `>5.5`
(12 matches), P[1] `>=10.0` → `>=6.5` (6) / `<3.5` → `<3.8` (5).
- B10 f_calc_28 vacuous predicate: :5871-5882 — P[1] `<600K` → `<1.5M` (7)
/ `>=20M` → `>=7M` (7). Plus F_CALC_30 P[0] retune collateral on new slice.
[verified and removed 2026-05-12]
- B1 f_writer_17__find_replace: libreoffice_writer.py:4130-4131 changed
`_old='managers'` → `'Managers'` + `_new='Team leads'` (case now matches
source para-3 opener). Smoke: pre-fix gold==source (vacuous), post-fix
1 substitution + 0 bytes match. Pre-fix `--max-turns 0` returned 1.0
baseline confirmed.
- B7 f_writer_21__font_georgia_p0 + f_writer_91__doc_font (and the
16-entry full-sweep block below): cleared by same 3-emitter fix as
the 5 validation doc_font entries above.
[verified and removed 2026-05-12] 4 VLC bugs:
- B3 f_vlc_14__rotate_90: vlc.py:391-433 `_av_mp4_src` now applies a
`drawtext` overlay (color name + frame number) on top of the lavfi
color source. pHash dist green-src vs rotated: 0 → **34** (>5 threshold).
- B12+B13 audio cluster (f_vlc_9 mp3/ogg + f_vlc_10 purple): vlc.py:391-433
added `sine_freq=` param. F_VLC_9 = 440 Hz, F_VLC_10 = 880 Hz. sha256
red.aac vs purple.aac = `4ae72c…` vs `0c47f3…` differ (was identical).
[verified and removed 2026-05-12]
f_vlc_16__trim_5s — eval/metrics.py::compare_videos_full_duration helper
added (ffprobe/OpenCV duration check with 0.5s tolerance; defers to
compare_videos with max_frames_to_check=600 so the 4s cap no longer hides
the duration mismatch). vlc.py rewires both params to the new func.
Oracle replay both params 0.0→1.0.
[verified and removed 2026-05-12]
B6+B16 f_tb_31__set_attr_a: thunderbird.py:975-991 + :1588-1605 — pre-config
now writes target xulstore widths directly (folderPaneBox=350, threadPaneBox=600).
Instruction reframed to observe-mode (verify the pre-sized pane, no drag).
Oracle replay both 0001/0002 validation. Recorded trajectory is obsolete
(was splitter-drag) — needs fresh rollout post-regen.
[verified and removed 2026-05-12] 7 vs_code marketplace
install entries (python/autodocstring/gitlens/cpptools/rust_analyzer/
jupyter/yaml) cleared via Option (d) in vs_code.py — synthetic-extension
pre-stage. `_make_extension_install_template` refactored: init purges
`~/.vscode/extensions/<id>-*` + drops matching entry from `extensions.json`;
oracle writes a minimal `package.json` + merges a registry entry whose
shape was captured live from VS Code 1.95.3. `code --list-extensions`
now reads the registry directly — no marketplace network round-trip.
7/7 oracle PASS (validate.py 75s, non-trivial pre-oracle confirmed);
docker probe: grep matches for all 7; max-turns 0 = 0.0 on python.
[verified and removed 2026-05-12] 30 photo_to_docx
entries (15 topics × {_photo_to_docx, _photo_to_docx_cover}) cleared by
the C2 fix in eval/metrics.py: `compare_docx_strict examine_images=True`
now decodes embedded images via PIL and pair-matches by SSIM ≥ 0.85
(PSNR/40dB fallback), so LO's JPEG re-encode on Ctrl+S no longer breaks
the byte-hash. Negative control (uniform fill) SSIM≈0.12 still rejects.
Quality sweep confirms threshold is well-calibrated for LO round-trip.
[verified and removed 2026-05-12] 16 writer
doc_font entries cleared by the same B7 3-emitter fix
(_docx_body_py:447 + _docx_structured_py:1255 + _src_qa_format:1466 now
emit `doc.styles['Normal'].font.name = font_name`).
[verified and removed 2026-05-12] 6 VLC
snapshot/wallpaper entries cleared by the same `_av_mp4_src` drawtext-overlay
fix — each frame is now visually unique (color name + frame number),
SSIM/compare_images can distinguish snapshot@t=N vs t=M. Verified: purple
snapshot t1 vs t4 pHash distance = 6 (was 0). F_VLC_23 wallpaper inherits
the same `_av_mp4_src` source upgrade.
[verified and removed 2026-05-12] 10 impress
strict-tolerance entries cleared by 3 factory edits in
libreoffice_impress.py: `_make_image_resize_filetask_template`,
`_make_image_size_filetask_template`, `_make_position_filetask_template`
now pass `approximately_tolerance=0.1` (10% relative) into the upstream
`compare_pptx_files.is_approximately_equal()` check. Oracle replay
d_imp_57__add_image_at_size 0.0→1.0 (attempt 1/2).
[verified and removed 2026-05-12]
f_chrome_34__search_united_flight P[1]: chrome.py:2186-2207 — gold_query
now includes `d2: "2026-06-17"`; instruction includes "returning 2026-06-17".
Oracle replay both 0001/0002 validation. Recorded trajectory obsolete (needs
fresh rollout post-regen).
---- impress (ambiguous — per user, temporarily on the bug list) ----
validation verification: release-notes infobar is present in turn_00 of
impress FN but does NOT overlap agent click targets (slide canvas at
y>375 vs infobar at y∈[155,225]). In 3 sampled FN traces, agent in
d_imp_01_0001 reached text-edit mode at turn_06 with infobar still
visible. The original "69-FN cascade-clear" estimate was substantially
inflated. xcu-seed fix MAY help writer `_p0` failures more than impress.
No specific FileTask added here — defer to measure-delta-post-fix.
