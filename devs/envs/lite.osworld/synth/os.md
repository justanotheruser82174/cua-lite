# OS — Synth Plan

> Keep in sync with code. Implementation: [`os.py`](/lite/gym/envs/lite/osworld/src/gen/train/synth/os.py).
> Common workflow: [`AGENTS.md`](/devs/envs/lite.osworld/synth/AGENTS.md). Cross-reference: [`perturb/os.md`](/devs/envs/lite.osworld/perturb/os.md).

## Current quant-gap snapshot (`measure_gap.py` v2)

Run `uv run python devs/envs/lite.osworld/measure_gap.py --domain os` for live numbers. Synth N=139, eval N=19 (5 infeasibility filtered). Post-cycle-47 expansion.

| Dim | Synth | Eval | Δpp | Status | Bridge |
|---|---:|---:|---:|:-:|---|
| `instruction_style.backtick_leak` | 0% | 0% | 0 | ✓ | **CLOSED** cycle-47. All 89 backtick-leak instructions rewritten in user voice (no backticks / no `(e.g., \`...\`)` hints). Each rewrite distinct. |
| `system_target.userspace_desktop` | 39.6% | 10.5% | +29 | 🔴 | **PARTIALLY CLOSED** (44 → 29). 16 new non-Desktop Files (`F_OS_38..F_OS_56`) added under `/tmp/` (NL2Bash multistep) and as `gsettings`/`dconf`/`xfconf-query` system-state rows. Remaining 🔴 is structural — most file_edit Files (F_OS_01..F_OS_25) are still anchored to Desktop. Further reduction requires pruning Desktop-anchored Files or relocating to `/etc/` (deferred — risk of breaking eval-path alignment). |
| `system_target.userspace_dotfile` | 12.9% | 0% | +13 | ⚠️ | Reduced 18→13pp by adding more non-dotfile rows (denominator increased from 101 → 139). |
| `skill_scope.file_edit` | 48.2% | 42.1% | +6 | ⚠️ | Reduced 15→6pp by adding gui_settings + shell_pipeline rows. |
| `skill_scope.gui_settings` | 30.2% | 36.8% | -7 | ⚠️ | Reduced -15→-7pp by adding 10 new GUI Settings Files (F_OS_38..F_OS_47) covering Universal Access / Sound / Region / Power / Notifications / Appearance / GNOME wallpaper / timezone. All evals read real system state. |
| `system_target.gsettings_dconf` | 21.6% | 21.1% | +1 | ✓ | **CLOSED** cycle-47. Added 6 new gsettings tasks (text-scaling, notifications, idle-delay, clock-format, color-scheme, high-contrast, screen-keyboard). |
| `system_target.other` | 17.3% | 21.1% | -4 | ✓ | **CLOSED** cycle-47. 9 new Files (F_OS_48..F_OS_56) live under `/tmp/` and route to `other`. |
| `system_target.sys_daemon` | 7.2% | 15.8% | -9 | ⚠️ | Added 3 new sys_daemon rows (mic-mute via pactl, timezone Asia/Tokyo via /etc/timezone, GNOME wallpaper via gsettings). Eval `pactl`/`gsettings`/`timedatectl` covered. |
| `skill_scope.shell_pipeline` | 21.6% | 21.1% | +1 | ✓ | Held aligned via 9 new NL2Bash multistep templates (F_OS_48..F_OS_56). |
| `difficulty_nl2bash.multistep_bash` | 21.6% | 21.1% | +1 | ✓ | Aligned — backtick strip dropped multistep coverage from 21 → 14 transiently; the 9 new NL2Bash-style templates restore it to 21.6%. |
| Package install / user account mgmt | 0% | 2/19 | -10 | ❌ | Deferred — would require new templates for `apt install` / `useradd` (eval `94d95f96`, `5812b315`). Lower priority than the above closures. |

**Cycle-47 expansion**: 16 new Files added (10 GUI Settings + 9 NL2Bash multistep + 6 sys_daemon helpers). Total File count 37 → 56; total FileTask count 45 → 61; total rows 101 → 139. Every new instruction is first-person user voice with NO backticks and NO `(e.g., \`...\`)` command leaks.

## Current shape

**56 Files (`F_OS_01`..`F_OS_56`) · 61 FileTask templates · 147 current jsonl rows after rescaling** (older snapshot: ~139).

Per-template row count = `min(2, distinct_param_count)`; most tasks ship 2 Params, then Stage B of the cross-domain scaler downgrades to fit the global budget. With `TARGET=math.inf` Stage B is a no-op.

| eval_class | count | native eval `func` |
|---|---|---|
| `file_operation` | 37 | `compare_text_file` / `check_include_exclude` / `compare_archive` (selected per-Param via `eval_kind`) |
| `system_query` | 2 | `compare_text_file` (file_diff over disk-usage / count probe output) |
| `dual_operation` | 2 | `compare_text_file` (file_diff over chmod-then-count / copy-then-count output) |
| `check_gnome_favorite_apps` | 2 | `check_gnome_favorite_apps` |
| `is_utc_0` | 1 | `is_utc_0` |
| `check_moved_jpgs` | 1 | `check_moved_jpgs` |

The first three buckets rebuild evaluators through `_build_evaluator(eval_kind, eval_args)` (`file_diff` / `command_output` / `config_check` / `archive_check`). The last three route directly to native OSWorld eval funcs.

## Architecture / design notes

**Eval files/task ratio**: 0.29. OS eval revolves around shell commands + filesystem state + GNOME settings.

**Key state-axis variation**: filesystem state (testDir layout: empty / mixed perms / nested); initial env vars / .bashrc / .zshrc / profile state; GNOME settings (default favorites, theme, timezone); config-file shapes (nginx / systemd / cron / sshd / hosts); terminal CWD.

**Files × tasks (current `os.py` `FILE_TASKS`).** Cap-2 per File enforced at emit time. F_OS_09 / F_OS_10 carry no tasks (compare_archive dropped).

| Loop | Files | Tasks | eval_class |
|---|---|---|---|
| 1 — real-OSS-style config files | F_OS_01 nginx_site_conf · F_OS_02 systemd_unit · F_OS_03 crontab · F_OS_04 sshd_config | 8 (2 per File) | `file_operation` × `config_check` |
| 2 — shell-rc / env-file shapes | F_OS_05 .bashrc · F_OS_06 .zshrc · F_OS_07 profile.sh · F_OS_08 .env | 8 | `file_operation` × `config_check` |
| 3 — archive / photo shapes | F_OS_09 / F_OS_10 carry 0 tasks · F_OS_11 tar_archive_extract (extract_to_subdir + extract_then_count) · F_OS_12 zip_photos_dir (list_jpgs_to_file) | 3 | `file_operation` × `command_output` |
| 4 — config-file variants v2 | F_OS_13 cron_schedule_v2 · F_OS_14 systemd_timer · F_OS_15 systemd_unit_b · F_OS_16 nginx_site_b | 8 | `file_operation` × `config_check` |
| 5 — file-tree / fs-state shapes | F_OS_17 mixed_filetree (chmod_644_recursive + make_subdir_archive) · F_OS_18 renameable_dir (rename + make_dated_subdir) · F_OS_19 authorized_keys (append + edit) · F_OS_20 .gitconfig (set_user + add_section) | 8 | `file_operation` |
| 6 — native-func mirrors + gap-fillers | F_OS_21 timezone_system (switch_to_utc → `is_utc_0`) · F_OS_22 gnome_favorites_real (remove_vim + add_terminal → `check_gnome_favorite_apps`) · F_OS_23 disk_query_dir (count_files + report_disk_usage → `compare_text_file`) · F_OS_24 dual_filetree (chmod_and_count + copy_and_count → `compare_text_file`) · F_OS_25 hosts_file (add_dev_host + swap_localhost_alias) · F_OS_26 photos_tree (copy_jpgs_to_flat_dir → `check_moved_jpgs`) | 8 | mixed |

**Active eval mix per `_build_evaluator`**:

| `eval_kind` (Param-level) | native `func` | typical FileTask types |
|---|---|---|
| `file_diff` | `compare_text_file` | file overwrites, ls/du/count → file, dual_operation outputs |
| `command_output` | `check_include_exclude` | shell-probe over stdout (`grep`, `cat`, `find`) |
| `config_check` | `check_include_exclude` | grep over config file body (nginx / systemd / sshd / bashrc / hosts) |
| `archive_check` | `compare_archive` | tar/zip-create flows (currently routed through ls-jpgs, not raw archive create) |
| `is_utc_0` | `is_utc_0` | `F_OS_21.switch_to_utc` |
| `check_gnome_favorite_apps` | `check_gnome_favorite_apps` | `F_OS_22.*` |
| `check_moved_jpgs` | `check_moved_jpgs` | `F_OS_26.copy_jpgs_to_flat_dir` |

**Gold-builder vocabulary**: `_gold_append_line` · `_gold_replace_line` · `_gold_chmod_recursive` · `_gold_rename_dir` · `_gold_tar_create` · `_gold_zip_create` · `_gold_tar_extract` · `_gold_make_subdir` · `_gold_ls_jpgs_to_file` · `_gold_extract_then_count` · `_gold_overwrite` · `_gold_set_timezone_utc` · `_gold_gsettings_favorites` · `_gold_copy_jpgs_recursive` · `_gold_du_to_file` · `_gold_count_files` · `_gold_chmod_then_count` · `_gold_copy_then_count`. The timezone / favorites / copy-jpgs builders ignore `src_path` and mutate real system state — kept for FileTask gold-callable signature uniformity.

## Implementation references

- `os.py` — `File` / `Param` / `FileTask` dataclasses; `_build_evaluator(eval_kind, eval_args)` dispatcher with native-func branches.
- All rows ship `synth_command=""` and use `config_override` to bypass the default app-launch (OS tasks are shell-only).
- [AGENTS.md §Scaler architecture](/devs/envs/lite.osworld/synth/AGENTS.md#scaler-architecture-cycle-41--design-5).
- [AGENTS.md §Per-domain Cat 1 / Cat 2 allocation guidance](/devs/envs/lite.osworld/synth/AGENTS.md#per-domain-cat-1--cat-2-allocation-guidance) — Cat 1 30% / Cat 2 70%; gap skills are config-file edits + .bashrc/.zshrc + archive extract + timezone switch + dual-operation compounds.

## Bridge plan / outstanding work

The quant snapshot is the canonical bridge plan; items it does not cover:

- Reject systemd-user-service / xdg-mime / cron-install variants without eval-task citation — re-grep eval to confirm before investing budget (fabrication risk).

## Cycle-recurring failures to avoid (os-specific)

- **F4 (CWD assumption).** Always use absolute paths OR force CWD via `.bashrc` auto-cd.
- **F2 (oracle artifact).** For `check_include_exclude` tasks, prefer `vm_command_line` (computed at eval time) over `cat <oracle-written-file>`.
- **File ownership (root vs user).** Flask `/execute` runs as root; if pre-config writes + chmod 600, user can't read. Always `chown user:user` after writing.
- **F9 (multi-step audio / GNOME UI).** For Cat 1, prefer command-line equivalents (`gsettings` / `timedatectl` directly instead of Settings UI navigation).
- **Trivial-pass on archive flows.** Never pre-create the destination archive (`tar.gz` / `zip`) in pre_config; the oracle must materialize it.

## Pipeline reference

`pre_config_steps` writes initial config/source files via heredoc + sets system state (`gsettings`, `/etc/timezone`); the agent acts in terminal/UI; eval runs `vm_command_line` probes or native funcs (`is_utc_0`, `check_gnome_favorite_apps`, `check_moved_jpgs`) directly.
