# VLC — Synth Plan

> Keep in sync with code. Implementation: [`vlc.py`](/lite/gym/envs/lite/osworld/src/gen/train/synth/vlc.py).
> Common workflow: [`AGENTS.md`](/devs/envs/lite.osworld/synth/AGENTS.md). Cross-reference: [`perturb/vlc.md`](/devs/envs/lite.osworld/perturb/vlc.md).

## Current quant-gap snapshot (`measure_gap.py` v2)

Run `uv run python devs/envs/lite.osworld/measure_gap.py --domain vlc` for live numbers. Synth N=62, eval N=15 (2 infeasibility filtered).

| Dim | Synth | Eval | Δpp | Status | Bridge |
|---|---:|---:|---:|:-:|---|
| `instruction_leak.menu_path_leak` | 0% | 0% | 0 | ✓ | Cycle-46 pass: 26 instructions rewritten to first-person intent (drop "via Tools → Preferences", "Video → Take Snapshot", "Media → Convert/Save", "Effects and Filters → Geometry"). Verified zero menu-path patterns post-fix. |
| `skill_class.prefs_vlcrc` | 32.3% | 53.3% | -21 | 🔴 | Cycle-46: added F_VLC_2X/3X/4F/4G — alt-init Files reusing existing `check_qt_max_volume` / `is_vlc_recordings_folder` / `check_qt_slider_colours` / `check_qt_bgcone` evaluator funcs (verified upstream). +7 vlcrc rows (13→20). Residual gap = eval is unusually heavy on prefs (53%); further bool-key Files blocked by trivial-pass constraint. |
| `media_source.remote_url_media` | 6.5% | 46.7% | -40 | 🔴 | Cycle-46: F_VLC_24 (Blender bigbuckbunny + sintel mp4) + F_VLC_27 (archive.org + samplelib mp3) → 4 HTTP-URL rows. URLs verified HTTP 200 at synth-gen time. |
| `media_source.hls_m3u8` | 3.2% | 6.7% | -3 | ✓ | Cycle-46: F_VLC_25 (Apple BipBop, mirrors upstream `bba3381f`) + F_VLC_26 (Mux test-streams) → 2 HLS rows. URLs verified HTTP 200. |
| `skill_class.file_m3u` | 0% | 0% | 0 | ✓ | Cycle-46 DROP: F_VLC_5/6/7 m3u FileTasks removed (6 rows). Eval has zero `check_list` coverage so the bucket is dead-weight; row budget redirected to prefs_vlcrc + remote_url. File instances kept for future re-introduction. |

## Current shape

**25 Files / 30 FileTasks → 37 current jsonl rows** (historical cycle-46 shape was 62 rows after scaling: dropped 3 m3u FileTasks, added 4 vlcrc Files + 5 vlcrc FileTasks + 4 remote-URL Files + 4 remote-URL FileTasks).

No real-asset bundle: `assets/synth/` is reserved for downloadable real data, but all VLC media is regenerable from code (ffmpeg lavfi `testsrc` / `sine` / `color`), so PD (3a) real-source ratio is **NA** for this domain (structurally exempt).

| setup_class | Files | eval_class | FileTasks | Eval `func` |
|---|---|---|---|---|
| `vlcrc_shape` | 12 (F_VLC_1, 2, 2X, 3, 3X, 4, 4B, 4C, 4D, 4E, 4F, 4G) | `config_setting` | 13 | `check_qt_bgcone`, `check_qt_max_volume`, `is_vlc_recordings_folder`, `check_qt_minimal_view`, `check_qt_slider_colours`, `check_global_key_play_pause`, `check_play_and_exit`, `check_one_instance_when_started_from_file` |
| ~~`m3u_playlist`~~ | ~~3 (F_VLC_5, 6, 7)~~ | ~~`check_list`~~ | 0 (cycle-46 DROP — eval has 0% file_m3u) | — |
| `ffmpeg_av_mp4` (with sine audio) | 2 (F_VLC_9, 10) | `media_transform` | 3 | `compare_audios`, `compare_images` |
| `ffmpeg_v_mp4` (video only) | 3 (F_VLC_13, 14, 16) | `media_transform` | 5 | `compare_images`, `compare_videos` |
| `live_media` / `live_media_srt` | 5 (F_VLC_17, 18, 19, 20, 21) | `is_vlc_playing` / `is_vlc_fullscreen` | 9 | `is_vlc_playing`, `is_vlc_fullscreen` |
| `remote_http_video` / `remote_hls_stream` / `remote_http_audio` | 4 (F_VLC_24 mp4, F_VLC_25/26 m3u8, F_VLC_27 mp3) | `is_vlc_playing` | 4 | `is_vlc_playing` (rule.type=url) |

Current generated total: **37 rows** in `train.synth.jsonl`.

Each FileTask exposes 2 Params (cap = `SYNTH_CAP_PARAMS_PER_TASK`), structurally distinct per PD (3b) — for live-state rows the second Param targets a `alt_basename` planted on Desktop, not a paraphrase clone.

## Architecture / design notes

**Eval files/task ratio**: 0.47. VLC eval mixes vlcrc preferences + media file ops + playback state.

**Eval evaluator-func mix**:
- vlcrc keys — `check_qt_bgcone`, `check_qt_max_volume`, `check_qt_minimal_view`, `check_qt_slider_colours`, `check_global_key_play_pause`, `check_play_and_exit`, `check_one_instance_when_started_from_file`, `is_vlc_recordings_folder`.
- Media transform — `compare_videos`, `compare_audios`, `compare_images`.
- Runtime state — `is_vlc_playing`, `is_vlc_fullscreen`.
- Playlist — `check_list` (rule-pattern match over a saved `.m3u`).

**Key state-axis variation**: vlcrc initial state (opposite-value seed planted by `_vlcrc_setup_step`); source media format (mp4 with/without sine audio, mp3 via ffmpeg lavfi); source media duration (live-state rows clamped ≥60s; transform rows kept short for ffmpeg speed); playlist track count (2 / 3 / 4 tracks).

**Mechanism summary**:

- **vlcrc rows** — pre_config writes the OPPOSITE value via `_vlcrc_setup_step`; oracle replays the target via `sed`; `_VLCRC_POSTCONFIG` kills VLC and relaunches so the running process re-reads the just-edited vlcrc.
- **m3u rows** — `_playlist_src` stages ffmpeg sine-wave mp3 tracks + a placeholder `.m3u` with a wrong entry; oracle writes the gold `.m3u`; eval anchors on `check_list` rule-pattern matching.
- **ffmpeg media-transform rows** — `_av_mp4_src` builds a deterministic source mp4 (lavfi color / sine); per-eval-kind extras (`_build_snapshot_extras` / `_build_media_compare_extras`) build a gold sink at `/tmp/gold_<template_id>.<ext>`. Oracle = `cp gold agent`; pre_config never creates the agent's sink path (trivial-pass guard).
- **live-state rows** — `_live_media_src` (and `_live_media_with_srt_src` for the lecture file) plant a long-enough (≥60s) media file under `~/Desktop` and launch an empty VLC with HTTP iface so the eval getter can curl the status XML. Oracle pkill+relaunch with the file loaded (or `xdotool key f` for fullscreen). `--http-password password` matches the lite-runner getter (`eval/runner.py:1322`).

### FileTask plan summary

**Loop 1 — vlcrc preference shapes (`config_setting`)**: `set_bgcone` (`215dfd39`), `set_max_volume` (`9195653c`), `set_recordings_folder` (`8ba5ae7a`), `set_minimal_view` (`a5bbbcd5`), `set_slider_colours` (`d06f0d4d`), `clear_global_play_pause` (`386dbd0e`), `set_play_and_exit` (`5ac2891a`), `set_one_instance` (`f3977615`).

**Loop 2 — `.m3u` playlists (`check_list`)**: `save_morning_drive_m3u` (3 tracks), `save_study_session_m3u` (2 tracks), `save_podcast_queue_m3u` (4 tracks).

**Loop 3 — ffmpeg av mp4** (snapshot + audio-extract): `extract_audio_mp3` (red+sine 5s), `extract_audio_purple` (purple+sine 6s), `snapshot_t1_purple` (purple+sine 6s).

**Loop 4 — ffmpeg video mp4** (snapshot + rotate + trim): `snapshot_t2` (blue 5s), `snapshot_t4` (green 5s), `rotate_90` (green 5s), `trim_5s` (cyan 10s), `snapshot_t6_cyan` (cyan 10s).

**Loop 5 — live media + fullscreen + subtitle sidecar**: `play_local_video` / `fullscreen_sample` (F_VLC_17 sample.mp4 / sample_b.mp4), `play_local_video_demo` / `fullscreen_demo` (F_VLC_18 demo_clip.mp4 / alt), `play_local_audio` (F_VLC_19 song.mp3 / song_b.mp3), `fullscreen_movie` / `play_local_movie` (F_VLC_20 movie.mp4 / movie_trailer.mp4), `play_lecture_with_srt` / `fullscreen_lecture_with_srt` (F_VLC_21 lecture.mp4+.srt / lecture_part2.mp4).

`F_VLC_4B` `qt-slider-colours` uses `rule={"type": "match", "expected_qt_slider_colours": "<12-int semicolon string>"}` per upstream `check_qt_slider_colours` (raw vlcrc string, not a list).

## Implementation references

- `vlc.py` — `File` / `FileTask` / `Param` dataclasses; per-`eval_kind` extra-builders (`_build_snapshot_extras` / `_build_media_compare_extras`).
- `_ffmpeg_make_mp4_cmd` / `_av_mp4_src` / `_playlist_src` / `_live_media_src` — deterministic ffmpeg-lavfi sources.
- `eval/runner.py:1316` — lite-runner getter; it issues a single `curl -s --user :password` (`:1322`), so synth must use `--http-password password`.
- [AGENTS.md §Scaler architecture](/devs/envs/lite.osworld/synth/AGENTS.md#scaler-architecture-cycle-41--design-5).
- [AGENTS.md §Per-domain Cat 1 / Cat 2 allocation guidance](/devs/envs/lite.osworld/synth/AGENTS.md#per-domain-cat-1--cat-2-allocation-guidance) — Cat 1 30% / Cat 2 70%; synth gap is media-transform + live-playback / fullscreen / playlist build.

## Bridge plan / outstanding work

The quant snapshot is the canonical bridge plan; items it does not cover:

- 2 eval rows are infeasible (DRM `7882ed6e`, auto-adjust `cb130f0d`) — known unaddressable.

## Cycle-recurring failures to avoid (vlc-specific)

- **F9 (deep Preferences UI navigation)**: Tools → Preferences → Show settings: All → expand category. For Cat 1, prefer settings reachable via top-level menu.
- **`compare_videos` strictness**: video re-encodes have pixel-level differences. Emit gold via the same ffmpeg codec/preset as the source so a `cp` oracle is byte-identical.
- **Playback-timing race (F10)**: `is_vlc_playing` checks the status XML at a fixed time; if VLC takes ≥source-duration seconds to start, eval fires after EOF. Clamp source duration ≥60s.
- **HTTP password mismatch (silent-eval-crash)**: the lite runner getter makes exactly ONE attempt, `curl -s --user :password`. Always launch with `--http-password password`; `tests/gym/envs/lite/osworld/test_lite_osworld.py` fails any vlcrc writer or launcher that seeds `vlc` instead.
- **Audio launch flag**: `_oracle_play_local` must NOT pass `--no-audio` for `media_kind == "audio"` (mp3 / ogg / wav have no video stream).

## Pipeline reference

`pre_config_steps` writes vlcrc (`_vlcrc_setup_step`) or builds a media file via ffmpeg lavfi (`_ffmpeg_make_mp4_cmd` / `_av_mp4_src` / `_playlist_src` / `_live_media_src`), then launches VLC (empty for live-state rows; with the source file for media-transform rows). Agent acts via UI; oracle = `sed` rewrite (vlcrc) / `cp gold agent` (media) / `printf > agent.m3u` (playlist) / `pkill + relaunch` (live-state). Eval reads vlcrc directly OR runs `compare_*` on the agent's sink file OR curls VLC's HTTP status XML.
