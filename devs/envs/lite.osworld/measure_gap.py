"""
devs/envs/lite.osworld/measure_gap.py — v0 quantitative train-vs-eval gap.

Companion to the historical per-domain manual audit (originally tracked in
measure_gap docs, now retired). Per-domain dimension catalog; each dimension produces
a categorical distribution over synth and upstream eval rows; Δpp = synth%
− eval%.

Status flags:
    ✓   |Δ| < 5pp           aligned
    ⚠️  5 ≤ |Δ| < 15pp     drift
    🔴  |Δ| ≥ 15pp          large gap
    ❌  eval% > 0, synth% = 0  eval-only-zero-synth (coverage hole)

For each (domain, dimension, category) the script also checks against
MANUAL_TARGETS (historical manual audit). If the quant value matches the manual finding
within ±5pp the row is tagged "✓cal"; else "❌cal" — this drives the
calibration loop "tighten classifier until it reproduces manual".

Run:
    uv run python devs/envs/lite.osworld/measure_gap.py                    # all registered domains
    uv run python devs/envs/lite.osworld/measure_gap.py --domain os
    uv run python devs/envs/lite.osworld/measure_gap.py --domain os chrome
    uv run python devs/envs/lite.osworld/measure_gap.py --calibration      # only show cal-tagged rows

v0 scope: OS + chrome fully classified, others have eval_fn_family only.
Extension order per historical audit priority: writer (compare_docx_strict examine_*
decomposition is the SUPER-EVALUATOR; do NOT count it as one bucket), then
calc, impress, multi_apps, gimp, thunderbird, vlc, vs_code.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable

from lite.utils.path import project_root

REPO_ROOT = project_root()
SYNTH_PATH = REPO_ROOT / "lite/gym/envs/lite/osworld/data/train.synth.jsonl"
EVAL_PATH = REPO_ROOT / "lite/gym/envs/lite/osworld/data/eval.jsonl"


# ============================================================================
# Loaders
# ============================================================================

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def domain_of(row: dict) -> str:
    return row["metadata"]["others"].get("domain", "unknown")


def oracle_actions_of(row: dict) -> list:
    actions = row["metadata"].get("others", {}).get("oracle_actions", [])
    return actions if isinstance(actions, list) else []


def evaluator_of(row: dict) -> dict:
    return row["metadata"].get("evaluator", {})


def eval_func_of(row: dict) -> str:
    f = evaluator_of(row).get("func", "NONE")
    return "+".join(f) if isinstance(f, list) else f


# ============================================================================
# Cross-domain dimensions (applied to every domain)
# ============================================================================

def atom_count(row: dict) -> str:
    """Compound-evaluator multiplicity. `fn1+fn2+fn3` ⇒ 3-atom.

    Many upstream eval rows use compound evaluators (5×`check_mp3_meta`,
    3×`compare_image_text`, 4×`diff_text_file`, multi-slide pptx). Synth
    almost always produces single-atom evaluators — measure this directly.
    """
    f = eval_func_of(row)
    n = len(f.split("+")) if f else 0
    if n <= 1: return "atom_1"
    if n == 2: return "atom_2"
    return "atom_3plus"


def result_type(row: dict) -> str:
    """`evaluator.result.type` — the *retrieval channel* for ground truth.

    Different result.types under the same eval-fn name mean different skills:
    - `vm_terminal_output` vs `vm_command_line` (os: stdout-capture vs exec)
    - `vm_wallpaper` vs `vm_file` (vlc: snapshot→wallpaper vs file)
    - `googledrive_file`, `accessibility_tree`, `is_in_vm_clickboard`,
      `find_installed_extension_name`, `bookmarks`, `open_tabs_info`,
      `audio_in_slide`, `background_image_in_slide` — distinct side-effect
      targets invisible at eval-fn-name level.
    """
    res = evaluator_of(row).get("result")
    if isinstance(res, dict):
        return res.get("type") or "no_type"
    if isinstance(res, list) and res and isinstance(res[0], dict):
        return res[0].get("type") or "no_type"
    return "no_result"


def is_infeasibility(row: dict) -> bool:
    """Infeasibility tasks are an eval-only signal by AGENTS.md INFEASIBLE_CLAIM_TRAIN.
    Train must NEVER include them — including them would teach a 'give up' shortcut.
    So we filter them out before computing any gap dimension."""
    f = eval_func_of(row)
    return "infeasible" in f.lower()


# ----------------------------------------------------------------------------
# Cross-cutting style classifiers (catch what per-domain leak detectors miss)
# ----------------------------------------------------------------------------

_GENERIC_PATH_RE = re.compile(r"/(?:home|tmp|etc|var|opt|root|sys|proc)(?:/|\b)")
_GENERIC_BT_RE = re.compile(r"`[^`\n]{1,200}`")
_FIRST_PERSON_RE = re.compile(
    r"\b(?:I |my |me |I'm |I'd |I'll |I've |could you|can you|please|help me|let me|I want|I need|"
    r"I'd like|I would like|assist me|would you|for me\b)",
    re.IGNORECASE,
)
_KEYSTROKE_RE = re.compile(r"\bCtrl\s*[+\-]\s*[A-Z]\b|\bShift\s*[+\-]\s*[A-Z]+\b|\bAlt\s*[+\-]\s*[A-Z]\b")
_MENU_ARROW_RE = re.compile(r"\s(?:→|->)\s")


def path_leak(row: dict) -> str:
    """Absolute-path leak: instruction names `/home/.../`, `/tmp/...`, `/etc/...`.

    Per-domain classifiers (multi_apps tool_leak, os backtick_leak) miss this:
    e.g. `Open /home/user/Desktop/white_background.png in GIMP and fill the
    white background with green` is a path-leak that no current classifier
    flags. Eval reaches 4% globally; synth runs 20%+.
    """
    return "path_leak" if _GENERIC_PATH_RE.search(row["instruction"]) else "no_path"


def backtick_any(row: dict) -> str:
    """ANY backticked token in instruction — domain-agnostic.

    Per-domain backtick detectors are narrow:
    - os `backtick_leak` only catches commands like `\\`touch X\\``
    - vs_code `key_leak` only catches dotted JSON keys
    - multi_apps `tool_leak` only catches pdftk/convert/pandoc names
    None of them flag `\\`head -n 5\\``, `\\`sort -f\\``, `\\`report_v2.pptx\\``,
    `\\`print('hello')\\`` — yet eval has ZERO backticks anywhere. This catches
    the residue.
    """
    return "any_backtick" if _GENERIC_BT_RE.search(row["instruction"]) else "no_backtick"


def length_bucket(row: dict) -> str:
    """Instruction-length bucket. Captures verbosity drift (gimp synth is 4×
    longer than eval; os synth is 1.7× longer) that no per-domain dim measures.
    """
    n = len(row["instruction"])
    if n < 80: return "very_short_<80"
    if n < 150: return "short_80-150"
    if n < 250: return "medium_150-250"
    if n < 400: return "long_250-400"
    return "very_long_400+"


def voice(row: dict) -> str:
    """First-person user voice vs impersonal/imperative recipe.

    Eval is 71% first-person ("could you", "I'd like", "help me"); synth runs
    52% — 19pp gap globally. Existing per-domain dims tag specific leaks but
    don't directly measure voice register.
    """
    return "first_person" if _FIRST_PERSON_RE.search(row["instruction"]) else "impersonal"


def keystroke_leak(row: dict) -> str:
    """Keystroke hint leak (Ctrl+L, Shift+End, etc.). Eval 0%; synth tail
    rows still slip these in (especially gimp/writer)."""
    return "keystroke_leak" if _KEYSTROKE_RE.search(row["instruction"]) else "no_keystroke"


def menu_arrow_leak(row: dict) -> str:
    """`→` / `->` menu-path leak. Eval 0%; synth residue in impress/vlc/gimp."""
    return "menu_arrow" if _MENU_ARROW_RE.search(row["instruction"]) else "no_arrow"


# ----------------------------------------------------------------------------
# Loop-2 cross-cutting classifiers (mood, sentence count, pre_config complexity,
# filename mention, quoted-value specificity, sentence length per atom).
# ----------------------------------------------------------------------------

_POLITE_RE = re.compile(r"^\s*(?:could you|can you|would you|will you|please|help me|i need|i'd like|i want|i wonder|i'm wondering|hi[ ,]|hey[ ,]|hello[ ,])", re.IGNORECASE)
_SENT_END_RE = re.compile(r"[.!?](?=\s|$)")
_FILENAME_RE = re.compile(
    r"\b\w+\.(?:pdf|docx|xlsx|pptx|csv|txt|md|html?|json|xml|png|jpe?g|gif|svg|"
    r"mp3|mp4|wav|m4a|ogg|epub|zip|tar|gz|tgz|conf|cfg|ini|yaml|yml|"
    r"py|js|ts|sh|bash|sql|log|m3u|vsix|odt|ods|odp)\b",
    re.IGNORECASE,
)
_QUOTE_VALUE_RE = re.compile(r"['\"][\w\s\-./@&,:]{2,40}['\"]")


def mood(row: dict) -> str:
    """Imperative / polite_request / question.

    Eval is ~24% question and ~20% polite_request — synth is 69% bare
    imperative ("Set X to Y", "Make slide N bold"). Imperative-heavy synth
    teaches the model a flat-command register; eval is a softer help-desk mix.
    """
    s = row["instruction"]
    head = s.strip()[:100]
    if "?" in s[:200]:
        return "question"
    if _POLITE_RE.match(head):
        return "polite_request"
    return "imperative"


def sentence_count_bucket(row: dict) -> str:
    """Eval has 22% of tasks with 3-4 sentences (context + ask + caveat) and
    5% with 5+ — synth is 67% single-sentence terse asks. Multi-sentence
    instructions teach context-extraction, which is core to the help-desk
    register."""
    n = len(_SENT_END_RE.findall(row["instruction"])) or 1
    if n == 1: return "1_sent"
    if n == 2: return "2_sent"
    if n <= 4: return "3-4_sent"
    return "5plus_sent"


def pre_config_complexity(row: dict) -> str:
    """Number of pre_config steps. Eval averages 2-3 lightweight steps
    (mostly file open / app launch); synth averages 3-5 with many bulky
    inline python heredocs. Heavy pre_config indicates synth-side
    infrastructure that doesn't exist in eval — a deployment-skew risk."""
    cfg = row.get("metadata", {}).get("config", [])
    n = len(cfg) if isinstance(cfg, list) else 0
    if n == 0: return "0_steps"
    if n <= 2: return "1-2_steps"
    if n <= 5: return "3-5_steps"
    return "6plus_steps"


def filename_in_instruction(row: dict) -> str:
    """Eval 15%; synth 31% — synth over-mentions explicit filenames like
    `report.docx` / `data.xlsx`. Eval typically references files by intent
    ("this spreadsheet", "the document I'm editing") — synth's explicit
    naming is a stylistic leak."""
    return "has_filename" if _FILENAME_RE.search(row["instruction"]) else "no_filename"


def quoted_value(row: dict) -> str:
    """Eval 40%; synth 24% — eval frequently names specific target values
    in quotes ("Times New Roman", "Q4 Report"); synth uses them less, which
    indicates synth's slot-fills are bare identifiers vs eval's
    quote-anchored specifics."""
    return "has_quoted" if _QUOTE_VALUE_RE.search(row["instruction"]) else "no_quoted"


# ----------------------------------------------------------------------------
# Loop-3 cross-cutting classifiers: discourse style, solution complexity,
# evaluator-tuning, app-name self-reference per domain.
# ----------------------------------------------------------------------------

_ACTION_VERB_RE = re.compile(
    r"\b(?:open|create|set|change|enable|disable|install|update|save|export|"
    r"delete|remove|rename|move|copy|edit|add|insert|configure|switch|toggle|"
    r"turn|make|adjust|apply|convert|compress|extract|filter|sort|find|search|"
    r"navigate|bookmark|click|select|highlight|fill|draw|paste|cut|undo|redo|"
    r"start|stop|play|pause|record|browse|download|upload|share|export|import|"
    r"format|paginate|justify|align|sort|group|aggregate|pivot)\b",
    re.IGNORECASE,
)
_DOMAIN_APPNAME_PATTERNS: dict[str, re.Pattern] = {
    "gimp": re.compile(r"\bgimp\b", re.IGNORECASE),
    "vlc":  re.compile(r"\bvlc\b", re.IGNORECASE),
    "thunderbird": re.compile(r"\bthunderbird\b", re.IGNORECASE),
    "chrome": re.compile(r"\bchrome\b", re.IGNORECASE),
    "vs_code": re.compile(r"\bvs ?code\b|\bvisual studio\b", re.IGNORECASE),
    "libreoffice_calc":    re.compile(r"\blibreoffice\b|\bcalc\b", re.IGNORECASE),
    "libreoffice_writer":  re.compile(r"\blibreoffice\b|\bwriter\b", re.IGNORECASE),
    "libreoffice_impress": re.compile(r"\blibreoffice\b|\bimpress\b|\bpowerpoint\b", re.IGNORECASE),
}


def context_prefix(row: dict) -> str:
    """Eval has 36% of instructions starting with an explanatory context
    sentence ("I'm preparing a deck for tomorrow's standup. Help me…"); synth
    has 15%. Synth dives straight to the ask, eval often sets up motivation.
    Heuristic: ≥2 sentences AND first sentence contains no action verb.
    """
    s = row["instruction"]
    sents = re.split(r"[.!?]+\s+", s.strip())
    if len(sents) < 2:
        return "no_context"
    first = sents[0]
    return "explain_first" if not _ACTION_VERB_RE.search(first) else "no_context"


def oracle_action_complexity(row: dict) -> str:
    """Number of oracle_actions (the gold-path action sequence). Eval averages
    2-4 actions per task (81% in 2-4 bucket); synth has 42% in the 0-1 bucket —
    indicating many synth tasks have trivial / no-op oracle paths (e.g. pure
    `cp gold sink` shell-only oracles), which is a complexity mismatch."""
    n = len(oracle_actions_of(row))
    if n <= 1: return "0-1_actions"
    if n <= 4: return "2-4_actions"
    if n <= 10: return "5-10_actions"
    return "11plus_actions"


def evaluator_tuning(row: dict) -> str:
    """Eval uses `examine_*` / `ignore_*` evaluator options on ~7% of tasks
    (legitimate tolerance for round-trip drift); synth uses them on 24% —
    over-reliance on tolerance-tuning indicates synth is patching brittle
    eval contracts rather than producing comparable artifacts."""
    e = row.get("metadata", {}).get("evaluator", {})

    def check(o):
        if isinstance(o, dict):
            return any("examine_" in str(k) or "ignore_" in str(k) for k in o.keys())
        if isinstance(o, list):
            return any(check(x) for x in o)
        return False

    return "uses_examine" if check(e.get("options")) else "no_examine"


def app_name_self_ref(row: dict) -> str:
    """Per-domain: does instruction name the app it acts on (e.g. "GIMP",
    "VLC", "Thunderbird")?

    Real eval users often say "make the picture brighter" without naming
    GIMP. Synth says "In GIMP, make the picture brighter" — gimp synth
    96% mentions GIMP, eval 12%; vlc synth 90% vs eval 47%. App-name
    self-reference is an instruction-style leak that no per-domain
    classifier currently flags.
    """
    d = domain_of(row)
    pat = _DOMAIN_APPNAME_PATTERNS.get(d)
    if pat is None:
        return "_n/a"  # os domain has no single app to name
    return "names_app" if pat.search(row["instruction"]) else "no_app_name"


def state_mutation_class(row: dict) -> str:
    """Read-only inspection (eval ~7-9% in chrome/multi_apps) vs state-mutating
    (everything else). Synth has 0% inspection-only tasks — missing the
    "tell me what URL is open" / "what's in the clipboard" class of tasks."""
    actions = oracle_actions_of(row)
    n = len(actions)
    if n == 0: return "no_oracle_actions"
    for a in actions:
        if isinstance(a, dict) and a.get("type") in ("clipboard_set", "screenshot", "wait"):
            return "inspect_only"
    return "state_mutating"


def init_file_count(row: dict) -> str:
    """Number of distinct files staged on disk in pre_config (Desktop / tmp).
    Eval is heavy on `download` config-steps (no path in cmd → counts as 0);
    synth often inline-generates 2-5 files. Reflects how much "scenery"
    synth pre-stages vs eval's more minimal setup."""
    files: set[str] = set()
    for s in _config_steps(row):
        if not isinstance(s, dict): continue
        p = s.get("parameters", {})
        cmd = p.get("command", "")
        if isinstance(cmd, list): cmd = " ".join(str(x) for x in cmd)
        if not isinstance(cmd, str): continue
        files.update(re.findall(r"/(?:home/user|tmp)/[^\s'\"]+\.\w{1,5}", cmd))
    n = len(files)
    if n == 0: return "0_files"
    if n <= 2: return "1-2_files"
    if n <= 5: return "3-5_files"
    return "6plus_files"


_EVAL_CHANNEL_MAP = {
    # broader-than-result_type semantic channel categories
    "vm_file": "file_contents", "cache_file": "file_contents",
    "googledrive_file": "file_contents", "cloud_file": "file_contents",
    "vm_command_line": "shell_stdout", "vm_terminal_output": "shell_stdout",
    "list_directory": "shell_stdout",
    "vlc_config": "config_file", "vscode_config": "config_file",
    "chrome_color_scheme": "config_file", "chrome_font_size": "config_file",
    "default_search_engine": "config_file", "new_startup_page": "config_file",
    "enable_do_not_track": "config_file", "enable_safe_browsing": "config_file",
    "data_delete_automacally": "config_file", "profile_name": "config_file",
    "active_tab_url": "browser_state", "active_tab_url_parse": "browser_state",
    "active_url_from_accessTree": "browser_state",
    "active_tab_html_parse": "browser_state", "active_tab_info": "browser_state",
    "bookmarks": "browser_state", "open_tabs_info": "browser_state",
    "history": "browser_state", "cookies": "browser_state",
    "url_path_parse": "browser_state", "url_dashPart": "browser_state",
    "page_info": "browser_state",
    "accessibility_tree": "a11y_tree", "vlc_playing_info": "live_app_state",
    "vm_wallpaper": "screen_state", "vm_screen_size": "screen_state",
    "is_in_vm_clickboard": "clipboard",
}
def eval_channel(row: dict) -> str:
    """Where does the evaluator READ from? More semantic than result_type —
    groups raw types into 7 channels: file_contents, shell_stdout, config_file,
    browser_state, a11y_tree, live_app_state, screen_state, clipboard, other.
    Catches eval channels synth never uses (a11y_tree in tb, screen_state)."""
    return _EVAL_CHANNEL_MAP.get(result_type(row), "other_channel")


def oracle_action_signature(row: dict) -> str:
    """Set of distinct action types in oracle_actions. Synth is 100% `execute`;
    eval has `chrome_open_tabs`, `close_window`, `wait`, etc. that synth
    never uses."""
    actions = oracle_actions_of(row)
    types = sorted({a.get("type", "?") for a in actions if isinstance(a, dict)})
    if not types: return "_empty"
    if len(types) == 1: return types[0]
    if len(types) == 2: return "+".join(types)
    return "3+_types_mixed"


_TOPIC_PATTERNS = {
    "finance":   re.compile(r"\b(price|sales|revenue|cost|profit|tax|invoice|budget|cash|loan|stock|payroll|gdp|economy)\b", re.I),
    "medical":   re.compile(r"\b(patient|doctor|hospital|disease|drug|medical|symptom|health|clinic|prescription|tamiflu)\b", re.I),
    "academic":  re.compile(r"\b(student|course|lecture|professor|class|exam|grade|thesis|research|paper|journal|citation|colab)\b", re.I),
    "travel":    re.compile(r"\b(flight|hotel|airport|reservation|booking|trip|vacation|airline|cruise|rental car)\b", re.I),
    "food":      re.compile(r"\b(recipe|restaurant|cuisine|meal|cook|ingredient|cafe|coffee|menu|food|drink)\b", re.I),
    "photo_art": re.compile(r"\b(photo|picture|image|portrait|landscape|painting|gallery|exhibit|artist)\b", re.I),
    "tech":      re.compile(r"\b(github|repo|commit|branch|docker|server|api|sdk|python|colab|gpt)\b", re.I),
    "personal":  re.compile(r"\b(my friend|my family|my todo|my reminder|my note|my diary|my home|my favorite)\b", re.I),
}
def topic_cluster(row: dict) -> str:
    """Subject-matter cluster — what is the task ABOUT semantically? Eval
    over-weights academic/tech, synth over-weights personal/photo/food. This
    matters for IID: a model SFT'd on heavy "personal" framing may
    underperform on eval's academic-flavored tasks."""
    s = row["instruction"]
    hits = [name for name, pat in _TOPIC_PATTERNS.items() if pat.search(s)]
    if not hits: return "untagged"
    if len(hits) == 1: return hits[0]
    return "multi_topic"


def oracle_modality(row: dict) -> str:
    """GUI (mouse/keystroke) vs CLI (shell) modality in oracle path.

    Chrome: synth 66% gui+cli vs eval 23% / 0% gui_only vs eval 35%.
    Gimp/tb/os: synth 100% cli_only vs eval has 12-21% gui+cli mix. This
    means synth's gold path teaches the agent a shell-heavy approach
    where eval expects GUI navigation."""
    a = oracle_actions_of(row)
    has_gui, has_cli = False, False
    for act in a:
        if not isinstance(act, dict): continue
        t = act.get("type", "")
        p = act.get("parameters", {}) or {}
        cmd = p.get("command", "")
        if isinstance(cmd, list): cmd = " ".join(str(x) for x in cmd)
        if not isinstance(cmd, str): cmd = ""
        if t in ("key", "click", "type", "open", "drag", "scroll"):
            has_gui = True
        elif t in ("launch",):
            # launch is GUI-flavored (opens app for user interaction)
            has_gui = True
        elif t in ("shell", "bash"):
            has_cli = True
        elif t == "execute":
            if "xdotool" in cmd or "pyautogui" in cmd or "wmctrl" in cmd:
                has_gui = True
            else:
                has_cli = True
    if has_gui and has_cli: return "gui+cli"
    if has_gui:             return "gui_only"
    if has_cli:             return "cli_only"
    return "no_modality"


_AMBIG_RE = re.compile(
    r"\b(some|several|a few|appropriate|suitable|reasonable|good\s+\w+|nice|"
    r"properly|somewhere|something|anywhere|anything|similar|like that|"
    r"the right one|whatever|whichever)\b",
    re.IGNORECASE,
)
def task_ambiguity(row: dict) -> str:
    """Eval intentionally uses vague slot-fillers ("some way", "appropriate
    column", "a few rows") on 5-10% of tasks; synth is 1-2%. Vagueness teaches
    the agent to make judgment calls — synth's bare precision robs this
    learning."""
    return "vague" if _AMBIG_RE.search(row["instruction"]) else "precise"


# ============================================================================
# Dimension classifiers — OS
# ============================================================================

# --- os: instruction_style (backtick-command leak vs user voice) ---

_BACKTICK_LEAK_RE = re.compile(r"`[^`\n]{1,120}`")
# A weaker signal too: parenthetical command hint
_PAREN_CMD_HINT_RE = re.compile(r"\((?:e\.g\.|i\.e\.|like|run|use|via)[^)]{0,200}`[^`]+`")


def os_instruction_style(row: dict) -> str:
    instr = row["instruction"]
    if _PAREN_CMD_HINT_RE.search(instr):
        return "backtick_leak"
    if _BACKTICK_LEAK_RE.search(instr):
        return "backtick_leak"
    return "user_voice"


# --- os: skill_scope (gui_settings vs shell_pipeline vs file_edit) ---

# GUI Settings / system-state query tokens. Captures both GNOME settings
# (gsettings/dconf) and adjacent sys-daemon queries (pactl/xfconf/timedatectl)
# that upstream eval uses to verify Settings-style tasks.
_SYS_STATE_EVAL_TOKENS = (
    "gsettings", "dconf", "favorite_apps",
    "pactl", "pulseaudio", "amixer",
    "xfconf-query", "xfconf",
    "timedatectl", "systemctl", "loginctl", "hostnamectl",
    "/etc/timezone", "/etc/localtime",
)
_SYS_STATE_FUNC_TOKENS = ("gnome", "is_utc_0")


def _is_gui_settings(row: dict) -> bool:
    ev_str = json.dumps(evaluator_of(row))
    if any(t in ev_str for t in _SYS_STATE_EVAL_TOKENS):
        return True
    f = eval_func_of(row).lower()
    return any(t in f for t in _SYS_STATE_FUNC_TOKENS)


def os_skill_scope(row: dict) -> str:
    # Note: infeasibility rows are pre-filtered in main() per AGENTS.md.
    if _is_gui_settings(row):
        return "gui_settings"
    if os_difficulty_nl2bash(row) == "multistep_bash":
        return "shell_pipeline"
    if eval_func_of(row) == "check_moved_jpgs":
        return "shell_pipeline"
    return "file_edit"


# --- os: system_target — where the truth lives ---

_DOTFILE_RE = re.compile(
    r"/home/user/(?:\.bashrc|\.zshrc|\.profile|profile\.sh|\.gitconfig|"
    r"\.vimrc|\.ssh/|\.config/|\.aliases?)"
)


def os_system_target(row: dict) -> str:
    # Note: infeasibility rows are pre-filtered in main() per AGENTS.md.
    ev = evaluator_of(row)
    ev_str = json.dumps(ev)

    if any(t in ev_str for t in ("gsettings", "dconf", "favorite_apps")):
        return "gsettings_dconf"
    if any(t in ev_str for t in (
        "pactl", "pulseaudio", "amixer", "xfconf-query",
        "timedatectl", "systemctl", "loginctl", "hostnamectl",
    )):
        return "sys_daemon"
    if "/etc/" in ev_str:
        return "etc_system"
    if "/home/user/Desktop" in ev_str:
        return "userspace_desktop"
    if _DOTFILE_RE.search(ev_str):
        return "userspace_dotfile"
    if "bash eval.sh" in ev_str or "check_password.sh" in ev_str:
        return "shell_eval_script"
    return "other"


# --- os: difficulty_nl2bash ---
# NL2Bash heuristic: combine conjoined-verb signal AND single-verb-with-filter signal.
# Upstream NL2Bash tasks include both command variants ("find + chmod" and "compress files modified ago").

_NL2BASH_VERBS = (
    "compress", "archive", "extract", "find", "search", "list", "sort", "count",
    "copy", "move", "delete", "rename", "chmod", "chown", "tar", "zip", "grep",
    "filter", "recurs", "show", "display",
)
_NL2BASH_CONNECTIVES = (" and ", " then ", ", and ", "; ", ", then ", " into ")
_NL2BASH_FILTERS = (
    "modified", "larger than", "older than", "newer than", "matching",
    "containing", "with permission", "by size", "by date", "by name",
    "recursively", "under /", "under current", "days ago", "minutes ago",
    "hours ago", "more than", "less than",
    "all regular files", "all subdirectories", "all files matching",
    "each line", "each of", "to each", "of all",
)


def os_difficulty_nl2bash(row: dict) -> str:
    instr = row["instruction"].lower()
    n_conn = sum(instr.count(c) for c in _NL2BASH_CONNECTIVES)
    distinct_verbs = sum(1 for v in _NL2BASH_VERBS if v in instr)
    has_filter = any(f in instr for f in _NL2BASH_FILTERS)
    if (n_conn >= 1 and distinct_verbs >= 2) or (has_filter and distinct_verbs >= 1):
        return "multistep_bash"
    return "single_op"


# ============================================================================
# Dimension classifiers — chrome
# ============================================================================

_URL_RE = re.compile(r"https?://[^\s\"'\)>]+")


def chrome_url_leak(row: dict) -> str:
    return "url_leaked" if _URL_RE.search(row["instruction"]) else "url_implicit"


def chrome_slot_resolution(row: dict) -> str:
    """Synth pre-resolves the target URL inside config (launch chrome with URL,
    pre-stage bookmarks, etc.); upstream eval launches chrome blank and forces
    the agent to navigate. measure_gap docs chrome D 🔴 large."""
    cfg_str = json.dumps(row["metadata"].get("config", []))
    if _URL_RE.search(cfg_str):
        return "config_preresolved_url"
    return "agent_must_navigate"


def chrome_relative_time(row: dict) -> str:
    """Eval uses `rule_relativeTime` (e.g. "next Monday", "this month",
    "8 months later") for 8 rows; synth grounds all dates as hard-coded
    `2026-06-26` literals. Detects via expected.rules[*].type."""
    ev = evaluator_of(row)
    rules_containers = [ev.get("expected"), ev.get("options")]
    for c in rules_containers:
        if isinstance(c, dict):
            r = c.get("rules") or c.get("expected")
            if isinstance(r, dict):
                r = [r]
            if isinstance(r, list):
                for rule in r:
                    if isinstance(rule, dict) and rule.get("type") == "rule_relativeTime":
                        return "relative_time"
        elif isinstance(c, list):
            for rule in c:
                if isinstance(rule, dict) and rule.get("type") == "rule_relativeTime":
                    return "relative_time"
    return "fixed_value"


_CHROME_FN_FAMILY = {
    "is_expected_tabs": "tabs",
    "is_expected_active_tab": "tabs",
    "is_expected_active_tab_approximate": "tabs",
    "is_expected_url_pattern_match": "tabs",
    "is_expected_bookmarks": "bookmarks",
    "is_expected_search_query": "search_query",
    "compare_pdfs": "pdf_export",
    "compare_htmls": "page_state",
    "check_direct_json_object": "settings_json",
    "check_font_size": "settings_json",
    "match_in_list": "history_or_list",
    "is_in_list": "history_or_list",
    "check_history_deleted": "history_or_list",
    "is_cookie_deleted": "cookies",
    "is_shortcut_on_desktop": "shortcut",
    "is_added_to_steam_cart": "cart_state",
    "exact_match": "exact_match",
    "infeasible": "infeasible",
}


def chrome_eval_fn_family(row: dict) -> str:
    f = eval_func_of(row).split("+")[0]
    return _CHROME_FN_FAMILY.get(f, f"other:{f}")


# ============================================================================
# Dimension classifiers — multi_apps (apps-per-task + app-combination)
# ============================================================================
# Detect which GUI apps are invoked by the instruction. Use lowercased keyword
# matching with disambiguators. Terminal/shell counts as 'os' (matches synth's
# 42% terminal-only "multi_apps" misnomer).

_APP_INSTR_KEYWORDS = {
    "chrome":      (r"\bchrome\b", r"\bfirefox\b", r"\bbrowser\b",
                    r"\bbookmark", r"\btab\b", r"\bweb ?page\b"),
    "writer":      (r"\bwriter\b", r"libreoffice writer", r"\bdocument\b",
                    r"\.docx\b", r"\.odt\b"),
    "calc":        (r"\bcalc\b", r"libreoffice calc", r"\.xlsx\b",
                    r"\bspread ?sheet", r"\bworkbook\b"),
    "impress":     (r"\bimpress\b", r"libreoffice impress", r"\.pptx\b",
                    r"\bslides?\b", r"presentation"),
    "gimp":        (r"\bgimp\b", r"image editor"),
    "vlc":         (r"\bvlc\b", r"\bplay (?:the |this |that )?(?:audio|video|music|song)"),
    "thunderbird": (r"\bthunderbird\b", r"\bemail\b", r"\binbox\b",
                    r"\bmail ?box\b", r"\bsent folder\b", r"\battachment"),
    "vs_code":     (r"\bvs ?code\b", r"\bvisual studio code\b"),
    "files":       (r"\bnautilus\b", r"\bfile manager\b", r"\bfiles app\b"),
    "shell":       (r"\bterminal\b", r"`[^`]*\b(sed|cat|find|tar|zip|unzip|"
                    r"pdftk|pandoc|convert|ffmpeg|grep|awk|chmod|cp|mv|"
                    r"rm|mkdir|ls|tree|wc|sort|uniq|head|tail|xargs|git)\b",
                    r"\bshell\s*(?:script|command)?\b",
                    r"\bcommand[- ]?line\b", r"\bbash\b"),
    # NEW: cross-app surfaces eval uses that v1 missed (multi_apps coverage holes)
    "gdrive":      (r"\bgoogle ?drive\b", r"\bg ?drive\b", r"\bdrive\.google",
                    r"\bupload (?:to|the).*drive"),
    "git":         (r"\bgit (?:push|pull|clone|commit|add|status|log|diff)",
                    r"\bgithub\b", r"\bgit ?repo\b", r"\bgit-clone"),
    "gnome":       (r"\bgsettings\b", r"\bgnome[- ]?(shell|tweaks?)\b",
                    r"\bgnome.theme", r"\bset (?:the )?wallpaper"),
    "vim":         (r"\bvim\b", r"\.vimrc\b"),
}

# Eval-function → app inference. When the instruction doesn't name the app
# but the evaluator obviously targets one, infer the app from the eval-fn.
# Upstream eval often phrases tasks abstractly ("count documents") and uses
# compare_table — the app is calc.
_EVAL_FN_TO_APP = {
    "is_expected_installed_extensions": "chrome",
    "is_expected_url_pattern_match+check_direct_json_object": "chrome",
    "check_line_number":      "shell",
    "compare_conference_city_in_order": "writer",
    "compare_table":         "calc",
    "compare_csv":           "calc",
    "compare_docx_files":    "writer",
    "compare_docx_files_and_ignore_new_lines": "writer",
    "compare_docx_tables":   "writer",
    "compare_docx_images":   "writer",
    "compare_docx_strict":   "writer",
    "compare_font_names":    "writer",
    "compare_line_spacing":  "writer",
    "evaluate_colored_words_in_tables": "writer",
    "evaluate_strike_through_last_paragraph": "writer",
    "check_highlighted_words": "writer",
    "is_first_line_centered": "writer",
    "has_page_numbers_in_footers": "writer",
    "compare_subscript_contains": "writer",
    "check_italic_font_size_14": "writer",
    "check_tabstops":        "writer",
    "compare_unique_train_records": "writer",
    "contains_page_break":   "writer",
    "compare_references":    "writer",
    "compare_pptx_files":    "impress",
    "compare_pptx_files_color_tolerant": "impress",
    "compare_pdfs":          "pdf",
    "compare_pdf_images":    "pdf",
    "check_pdf_pages":       "pdf",
    "compare_epub":          "pdf",
    "compare_images":        "gimp",
    "compare_image_list":    "gimp",
    "compare_image_text":    "gimp",
    "check_image_file_size": "gimp",
    "check_image_size":      "gimp",
    "check_structure_sim":   "gimp",
    "check_structure_sim_with_threshold": "gimp",
    "check_brightness_decrease_and_structure_sim": "gimp",
    "check_mp3_meta":        "vlc",
    "compare_audios":        "vlc",
    "check_thunderbird_folder": "thunderbird",
    "check_thunderbird_filter": "thunderbird",
    "check_thunderbird_prefs":  "thunderbird",
    "is_expected_tabs":      "chrome",
    "is_expected_bookmarks": "chrome",
    "is_expected_search_query": "chrome",
    "is_expected_url_pattern_match": "chrome",
    "compare_htmls":         "chrome",
    "check_accessibility_tree": "chrome",
    "is_extension_installed": "vs_code",
    "check_python_file_by_test_suite": "vs_code",
    "diff_text_file":        "vs_code",
    "compare_python_pure_text": "vs_code",
    "check_json":            "vs_code",
    "compare_archive":       "shell",
    "compare_zip_files":     "shell",
    "check_list":            "shell",
    "check_include_exclude": "shell",
    "file_contains":         "shell",
    "exact_match":           "shell",
    "compare_text_file":     "shell",
    "literal_match":         "shell",
    "fuzzy_place_math":      "shell",
    "is_in_list":            "shell",
    "is_in_vm_clickboard":   "shell",
    "compare_result_files":  "shell",
}


def _apps_from_row(row: dict) -> set[str]:
    """Detect apps used: instruction keywords ∪ eval-fn inference."""
    instr = row["instruction"].lower()
    hit: set[str] = set()
    for app, patterns in _APP_INSTR_KEYWORDS.items():
        if any(re.search(p, instr) for p in patterns):
            hit.add(app)
    # Add apps from eval-fn signal (compound funcs split on '+')
    for sub in eval_func_of(row).split("+"):
        app = _EVAL_FN_TO_APP.get(sub)
        if app:
            hit.add(app)
    return hit


def multi_apps_per_task(row: dict) -> str:
    n = len(_apps_from_row(row))
    if n <= 1:
        return "apps_le_1"
    if n == 2:
        return "apps_2"
    return "apps_3plus"


def multi_apps_combination(row: dict) -> str:
    """Sorted '+'-joined label of apps used (instruction ∪ eval-fn)."""
    apps = _apps_from_row(row)
    if not apps:
        return "_none"
    if len(apps) == 1:
        return next(iter(apps))
    return "+".join(sorted(apps)[:3])


# --- multi_apps: tool_leak — backticked pdftk/pandoc/IM/ffmpeg in instruction.
# measure_gap docs multi_apps F 🟡 medium. Upstream names the app, not the flags.

_MULTI_APPS_TOOL_LEAK_RE = re.compile(
    r"`[^`\n]*\b(pdftk|pandoc|convert|magick|ffmpeg|imagemagick|"
    r"zip|unzip|tar|sed|awk|grep|jq|xmllint|csvkit|picard|kid3)\b[^`\n]*`",
    re.I,
)


def multi_apps_tool_leak(row: dict) -> str:
    instr = row["instruction"]
    return "tool_leak" if _MULTI_APPS_TOOL_LEAK_RE.search(instr) else "no_leak"


# ============================================================================
# Dimension classifiers — libreoffice_writer (compare_docx_strict examine_* decomp)
# ============================================================================
# CRITICAL (per /measure_gap docs writer section): compare_docx_strict is the right
# super-evaluator and covers many upstream funcs via examine_* flags.
# Do NOT treat compare_docx_strict as one bucket — decompose by examine_* flag
# and map onto upstream skill_class families.

# Writer skill_class taxonomy (post-cycle-46, examine_*-aware):
#
#   text_match           compare_docx_strict (default) ↔ compare_docx_files
#   tables               examine_tables / *_table_contents ↔ compare_docx_tables
#                                                           / evaluate_colored_words_in_tables
#   images               examine_images ↔ compare_docx_images
#   font_name            examine_font_name ↔ compare_font_names / find_default_font
#   line_spacing         examine_line_spacing ↔ compare_line_spacing
#   highlight            examine_highlight ↔ check_highlighted_words
#   font_size            examine_font_size ↔ check_italic_font_size_14
#   color                examine_color ↔ (no clean upstream peer)
#   pdfs                 compare_pdfs / check_pdf_pages
#   specialized_uncov    upstream-only narrow checkers with NO examine_* peer
#                        (is_first_line_centered, has_page_numbers_in_footers,
#                         compare_subscript_contains, evaluate_strike_through_last_paragraph,
#                         check_tabstops, compare_unique_train_records, contains_page_break)
#   infeasible

_WRITER_SPECIALIZED_UNCOV = {
    "is_first_line_centered",
    "has_page_numbers_in_footers",
    "compare_subscript_contains",
    "evaluate_strike_through_last_paragraph",
    "check_tabstops",
    "compare_unique_train_records",
    "contains_page_break",
}
_UPSTREAM_WRITER_FN_TO_SKILL = {
    "compare_docx_files": "text_match",
    "compare_docx_files_and_ignore_new_lines": "text_match",
    "compare_docx_tables": "tables",
    "evaluate_colored_words_in_tables": "tables",
    "compare_docx_images": "images",
    "compare_font_names": "font_name",
    "find_default_font": "font_name",
    "compare_line_spacing": "line_spacing",
    "compare_pdfs": "pdfs",
    "check_pdf_pages": "pdfs",
    "check_highlighted_words": "highlight",
    "check_italic_font_size_14": "font_size",
    "infeasible": "infeasible",
}


def writer_skill_class(row: dict) -> str:
    """Skill class — writer's eval-fn name carries NO skill info because synth
    uses `compare_docx_strict` for ~34% of rows (one fn, ≥8 different skills via
    `examine_*` flags). Decomposition is the ONLY way to match writer skills.

    Synth uses a DUAL pattern:
      - 66 / 196 rows (~34%): `compare_docx_strict` + at most one `examine_*=True`
        flag → decompose via the flag (or `text_match` if no flag set).
        Verified: 0 synth rows have ≥2 examine_* flags, so priority order is safe.
        Synth's examine_* vocabulary covers 5 properties: color / font_name /
        font_size / highlight / images. Missing (vs eval): tables / line_spacing
        / subscript / page_break / tabstops / centered / footer / strikethrough.
      - 130 / 196 rows (~66%): specific upstream-named fns directly
        (`compare_line_spacing` ×19, `compare_docx_tables` ×3, `compare_pdfs` ×4,
        `contains_page_break` ×10, `has_page_numbers_in_footers` ×8, etc.) →
        handled by _WRITER_SPECIALIZED_UNCOV / _UPSTREAM_WRITER_FN_TO_SKILL.
    """
    ev = evaluator_of(row)
    f = eval_func_of(row).split("+")[0]

    if f == "compare_docx_strict":
        opts = ev.get("options") or {}
        # Priority order on examine_* flags. Synth has 0 multi-flag rows
        # (verified), so first-match never collapses real multi-skill rows.
        if opts.get("examine_images"):    return "images"
        if opts.get("examine_font_name"): return "font_name"
        if opts.get("examine_highlight"): return "highlight"
        if opts.get("examine_font_size"): return "font_size"
        if opts.get("examine_color"):     return "color"
        # No examine_* set → strict byte-equality of the whole docx.
        # Synth intends "tests everything via byte match", not "tests text only".
        return "text_match"

    if f in _WRITER_SPECIALIZED_UNCOV:
        return "specialized_uncov"
    return _UPSTREAM_WRITER_FN_TO_SKILL.get(f, f"other:{f}")


def writer_evaluator_pattern(row: dict) -> str:
    """How is the row's skill being tested? Surfaces the synth dual pattern.

    - `compare_docx_strict+examine_*` — synth's super-evaluator with one
      examine_* flag (synth-only style; eval never uses compare_docx_strict).
    - `compare_docx_strict_default` — compare_docx_strict with NO examine_*;
      byte-strict equality of the whole doc (synth-dominant — agent learns
      "produce byte-identical docx", stricter than 14/22 upstream eval requires).
    - `specific_upstream_fn` — named upstream fn like compare_font_names,
      compare_docx_tables, is_first_line_centered (eval canonical; synth uses
      these for ~66% of rows).
    - `compound_multi_property` — `+`-joined evaluator (multi-property guard,
      e.g. compare_docx_files+compare_subscript_contains). Eval has 5; synth: 0.
    """
    f_raw = eval_func_of(row)
    f = f_raw.split("+")[0]
    if "+" in f_raw:
        return "compound_multi_property"
    if f == "compare_docx_strict":
        opts = evaluator_of(row).get("options") or {}
        has_flag = any(k.startswith("examine_") and v is True for k, v in opts.items())
        return "compare_docx_strict+examine_flag" if has_flag else "compare_docx_strict_default"
    return "specific_upstream_fn"


# --- writer: target_anchor (ordinal vs quote-anchored vs doc-wide) ---
# measure_gap docs writer 4: synth 44% ordinal, eval 26% ordinal + 13% quote-anchor.
# Synth never quote-anchors (`bold "the second sentence"`).

_ORDINAL_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|opening|last|final|nth|"
    r"1st|2nd|3rd|4th|5th)\s+"
    r"(?:paragraph|sentence|line|word|heading|chapter)",
    re.I,
)
_QUOTE_TARGET_RE = re.compile(r'"[^"\n]{2,40}"')


def writer_target_anchor(row: dict) -> str:
    instr = row["instruction"]
    has_quote = bool(_QUOTE_TARGET_RE.search(instr))
    has_ordinal = bool(_ORDINAL_RE.search(instr))
    if has_quote and has_ordinal: return "ordinal+quote"
    if has_quote:                 return "quote_anchor"
    if has_ordinal:               return "ordinal"
    return "doc_wide"


# ============================================================================
# Dimension classifiers — calc / impress / gimp / thunderbird / vlc / vs_code
# (one headline dimension each — see /measure_gap docs per-domain bridge plan)
# ============================================================================

# --- calc: save_protocol (open in config + Ctrl+S in postconfig) ---
# measure_gap docs calc P1: 266/266 synth lack `open`, 47/47 eval have `open`.

def _config_steps(row: dict) -> list[dict]:
    return row["metadata"].get("config", []) or []

def _postconfig_steps(row: dict) -> list[dict]:
    pc = evaluator_of(row).get("postconfig")
    return pc or []

_CTRL_S_PATTERNS = (
    "ctrl+s",                             # key step `"key": "ctrl+s"`
    'hotkey("ctrl", "s"',                 # pyautogui (double-quoted)
    "hotkey('ctrl', 's'",                 # pyautogui (single-quoted)
    'hotkey(\\"ctrl\\", \\"s\\"',         # pyautogui inside JSON-escaped string
)
def calc_save_protocol(row: dict) -> str:
    has_open = any(s.get("type") == "open" for s in _config_steps(row))
    pc_json = json.dumps(_postconfig_steps(row)).lower()
    has_ctrl_s = any(pat in pc_json for pat in _CTRL_S_PATTERNS)
    if has_open and has_ctrl_s:   return "open+ctrl_s"
    if has_open:                  return "open_only"
    if has_ctrl_s:                return "ctrl_s_only"
    return "neither_save_as_trap"


# --- impress: comparator_strictness (synth-only tolerant variants) ---
# measure_gap docs impress B: synth 29% color-tolerant + 6 position-tolerant; eval 0%.

def impress_comparator(row: dict) -> str:
    # Note: infeasibility rows are pre-filtered in main() per AGENTS.md.
    f = eval_func_of(row).split("+")[0]
    if f == "compare_pptx_files":                       return "strict"
    if f == "compare_pptx_files_color_tolerant":        return "color_tolerant"
    if f == "compare_pptx_files_position_tolerant":     return "position_tolerant"
    return f"other:{f}"


# --- gimp / thunderbird / vlc / vs_code: instruction-leak family ---
# Each domain has a characteristic engineer-recipe leak pattern in instructions:
#   gimp        — keystroke hints (Ctrl+L, Filters → Levels, etc.)
#   thunderbird — pref-key leaks (mail.*, browser.*, threadPaneBox)
#   vlc         — menu-path leaks (Tools → Preferences, Video → Take Snapshot)
#   vs_code     — backticked JSON-key leaks (`editor.wordWrap`, `files.autoSave`)
# All eval-side: never leak the specific token. Direction-of-gap is the signal.

_GIMP_LEAK_PATTERNS = (
    r"\bCtrl\s*\+\s*[A-Z]", r"\bShift\s*\+\s*[A-Z]",
    r"Filters?\s*[>→]\s*", r"Image\s*[>→]\s*", r"Tools?\s*[>→]\s*",
    r"Layer\s*[>→]\s*", r"Colors?\s*[>→]\s*",
    r"\b(?:via|using)\s+(?:the\s+)?(?:Levels?|Curves?|Hue.?Saturation|Threshold) dialog",
)
def gimp_instruction_leak(row: dict) -> str:
    instr = row["instruction"]
    return "leak" if any(re.search(p, instr, re.I) for p in _GIMP_LEAK_PATTERNS) else "no_leak"


_TB_PREF_KEY_PATTERNS = (
    r"`[a-z]+\.[a-z_.]+`",                  # any backticked dotted key
    r"\b(?:mail|browser|extensions|app|general)\.[a-z_.]+",
    r"\babout:config\b",
    r"\bthreadPaneBox\b", r"\b(?:user_pref|setBoolPref|setIntPref)\b",
)
def thunderbird_instruction_leak(row: dict) -> str:
    instr = row["instruction"]
    return "pref_key_leak" if any(re.search(p, instr, re.I) for p in _TB_PREF_KEY_PATTERNS) else "no_leak"


_VLC_MENU_PATH_PATTERNS = (
    r"\b(?:via|using)\s+(?:the\s+)?(?:Tools|Media|Video|Audio|View|Subtitle)\s*[>→]",
    r"\bTools\s*[>→]\s*Preferences", r"\bMedia\s*[>→]\s*Convert",
    r"\bVideo\s*[>→]\s*Take Snapshot", r"\b(?:Effects? and Filters?)\b",
)
def vlc_instruction_leak(row: dict) -> str:
    instr = row["instruction"]
    return "menu_path_leak" if any(re.search(p, instr, re.I) for p in _VLC_MENU_PATH_PATTERNS) else "no_leak"


_VS_CODE_KEY_PATTERNS = (
    r"`[a-z]+\.[a-zA-Z_.]+`",               # any backticked dotted key
    r"\b(?:editor|files|workbench|terminal|debug|python|extensions|window)"
    r"\.[a-zA-Z_.]+",
)
def vs_code_instruction_leak(row: dict) -> str:
    instr = row["instruction"]
    return "key_leak" if any(re.search(p, instr) for p in _VS_CODE_KEY_PATTERNS) else "no_leak"


# --- calc: skill_class — decompose `compare_table` by expected.rules.type ---
# measure_gap docs calc C 🔴 large: synth heavy on derived/filter/groupby; eval has
# pivot (5) + sheet_print (5) + style (2) etc. that compare_table-only counting
# can't see.

def _calc_table_rule_types(row: dict) -> list[str]:
    """Extract `compare_table` sub-rule types.

    v1 bug: lite eval.jsonl stores rules at `evaluator.options.rules`, not
    `evaluator.expected.rules`. Synth uses the same `options.rules` shape.
    Check BOTH paths so the classifier works on every row.
    """
    ev = evaluator_of(row)
    rules = None
    for container_key in ("options", "expected"):
        c = ev.get(container_key)
        if isinstance(c, dict):
            r = c.get("rules")
            if isinstance(r, list):
                rules = r
                break
            if isinstance(r, dict):
                rules = [r]
                break
        if isinstance(c, list):
            rules = c
            break
    if not isinstance(rules, list):
        return []
    return [r.get("type") for r in rules if isinstance(r, dict) and r.get("type")]


def calc_skill_class(row: dict) -> str:
    f = eval_func_of(row).split("+")[0]
    if f != "compare_table":
        return f or "unknown"
    types = _calc_table_rule_types(row)
    if not types:
        return "compare_table_other"
    return Counter(types).most_common(1)[0][0]


def calc_rule_combo(row: dict) -> str:
    """Sorted-set label of compare_table rule.types per row. Exposes SheetCopilot
    chained-skill shape (sheet_data+chart, pivot_table+check_cell, etc.).
    Synth has 181 multi-rule rows but only over {sheet_data, style, sheet_name};
    eval has 14/46 with chart/pivot/sheet_print mixes."""
    f = eval_func_of(row).split("+")[0]
    if f != "compare_table":
        return "_not_compare_table"
    types = _calc_table_rule_types(row)
    if not types:
        return "_empty"
    return "+".join(sorted(set(types)))


def calc_source_provenance(row: dict) -> str:
    """Did the config download a curated .xlsx, or inline-generate via openpyxl?
    Synth=98% inline, eval=100% curated. Biggest content gap in calc.
    Detectable via config-step types."""
    steps = row["metadata"].get("config", []) or []
    has_download = any(s.get("type") == "download" for s in steps)
    if has_download:
        return "curated_real_xlsx"
    has_exec_openpyxl = any(
        s.get("type") == "execute"
        and "openpyxl" in json.dumps(s.get("parameters", {}))
        for s in steps
    )
    if has_exec_openpyxl:
        return "synth_inline_openpyxl"
    return "other"


# --- impress: op_family — title-style vs background vs image vs structural ---
# measure_gap docs impress A 🔴 large: synth 50% title/body style vs eval 23%.

def impress_op_family(row: dict) -> str:
    instr = row["instruction"].lower()
    if any(k in instr for k in (
        "title", "bold", "italic", "underline", "font size", "make the title",
        "color of the title", "color of the heading", "size to", "alignment",
    )):
        return "title_or_body_style"
    if any(k in instr for k in ("background", "fill the slide", "slide fill", "color the slide")):
        return "background"
    if any(k in instr for k in ("image", "picture", "photo", "snapshot")):
        return "image"
    if "table" in instr and "slide" in instr:
        return "table"
    if any(k in instr for k in ("transition", "presenter", "slide number", "page number")):
        return "structural"
    if any(k in instr for k in ("save as", "export", ".pptx", ".png", ".pdf")):
        return "save_or_export"
    if any(k in instr for k in ("add slide", "duplicate slide", "summary slide", "reorder", "move slide")):
        return "slide_structural"
    return "other"


# --- impress: rgb_triplet_leak ---
# measure_gap docs impress C 🟡 medium: synth leaks `RGB 30,130,30` triplets in ~70%
# of color tasks; eval uses color names ("yellow", "dark red 2").

_RGB_TRIPLET_RE = re.compile(r"\bRGB\s*\(?\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}", re.I)


def impress_rgb_leak(row: dict) -> str:
    return "rgb_triplet_leak" if _RGB_TRIPLET_RE.search(row["instruction"]) else "no_rgb"


_SLIDE_TITLE_ANCHOR_RE = re.compile(
    r"(?:on |in |of |the )['\"]([^'\"]{2,40})['\"]\s+slide|"
    r"slide (?:named|titled|called)\s+['\"]?([^'\".\n]{2,40})['\"]?",
    re.I,
)
_SLIDE_ORDINAL_RE = re.compile(
    r"\bslide\s*\d+\b|"
    r"\b(?:first|second|third|fourth|fifth|last|final|opening)\s+slide\b",
    re.I,
)


def impress_slide_anchor(row: dict) -> str:
    """How does the instruction reference a target slide?

    `title_text_anchor` — e.g. "In the 'Features' slide..." (eval real-deck pattern)
    `ordinal_anchor`    — e.g. "on slide 3" / "the first slide" (synth dominant)
    `doc_wide`          — no specific slide referenced (background-all etc.)
    """
    instr = row["instruction"]
    if _SLIDE_TITLE_ANCHOR_RE.search(instr):
        return "title_text_anchor"
    if _SLIDE_ORDINAL_RE.search(instr):
        return "ordinal_anchor"
    return "doc_wide"


# --- thunderbird: async_flush_pattern ---
# measure_gap docs thunderbird 3 🔴 large: 30/67 rows lack `pkill -9 thunderbird` in
# postconfig → prefs.js / xulstore.json async-flush race → false negatives.

def thunderbird_async_flush(row: dict) -> str:
    """Postconfig shutdown protocol. v1 polarity correction:

    - Eval canonical is `close_window_only` (15/15 rows; uses profile snapshots,
      doesn't need force-kill).
    - Synth INVENTED `pkill_kill_signal` for ~30% of rows — this is the gap,
      NOT a bug-fix. The pkill is over-aggressive for the eval contract.
    - `no_postconfig` = rows with empty postconfig (no shutdown step at all).
    """
    pc_str = json.dumps(evaluator_of(row).get("postconfig", []) or []).lower()
    if not pc_str or pc_str == "[]":
        return "no_postconfig"
    has_pkill_tb = ("pkill" in pc_str) and ("thunderbird" in pc_str)
    has_close_window = "close_window" in pc_str
    if has_pkill_tb:    return "pkill_kill_signal"
    if has_close_window: return "close_window_only"
    return "other_postconfig"


_TB_PREF_KEY_FAMILIES = (
    (r"\bmail\.identity\.|\bsignature[_ ]text\b|\bsignature\b", "signature"),
    (r"\bmail\.html_compose\b|\bcompose\b.*\bhtml\b", "compose_html"),
    (r"\bjunk\b|\bmanualMark\b|\bspam\b|\bsafebrowsing\b", "junk_safebrowsing"),
    (r"\breply\b|\bquote\b|\bmailnews\.reply\b", "reply_quote"),
    (r"\bdark[ _\-]?mode\b|\bactiveThemeID\b|\btheme\b", "dark_theme_ui"),
    (r"\bapplyIncomingFilters\b|\binbox.*filter\b", "apply_filters"),
    (r"\bmail\.server\.|\bcheck_new_mail\b|\bauto[ _\-]?check\b", "server_check"),
    (r"\bthreadPaneBox\b|\bfolderPaneOpen\b|\buidensity\b|\bxulstore\b", "xulstore_ui"),
    (r"\binline_attachments\b|\battach\b", "attachment"),
    (r"\bmail\.mdn\b|\breport\b", "mdn_report"),
)


def thunderbird_pref_key_family(row: dict) -> str:
    """For prefs/xulstore rows, which pref-key family does it touch?

    Synth clusters on safebrowsing/html_compose; eval probes signature_text,
    applyIncomingFilters, dark-mode, mailnews.reply_quoting — different families.
    Look at instruction + evaluator JSON for the dotted key.
    """
    f = eval_func_of(row).split("+")[0]
    if f not in ("check_thunderbird_prefs", "check_json"):
        return "_not_pref_row"
    haystack = (row["instruction"] + " " + json.dumps(evaluator_of(row))).lower()
    for pattern, family in _TB_PREF_KEY_FAMILIES:
        if re.search(pattern, haystack, re.I):
            return family
    return "other_pref"


# --- vlc: media_source ---
# measure_gap docs vlc B 🟢 low: synth never uses remote URL / HLS streams.

def vlc_media_source(row: dict) -> str:
    """Local file vs remote URL vs HLS stream vs D-Bus-only (no media)."""
    cfg_str = json.dumps(row["metadata"].get("config", []) or [])
    instr = row["instruction"]
    if "m3u8" in cfg_str or "m3u8" in instr.lower():
        return "hls_m3u8"
    if re.search(r"https?://[^\s\"]+\.(mp4|mp3|webm|ogg|flv|avi)", cfg_str + " " + instr, re.I):
        return "remote_url_media"
    if re.search(r"https?://", cfg_str):
        return "remote_url_media"
    if re.search(r"/home/user/[^\s\"]*\.(mp4|mp3|webm|wav|flac|mkv|avi)", cfg_str):
        return "local_file"
    return "dbus_or_no_media"


# --- gimp: skill_class — by eval_fn ---
# measure_gap docs gimp B 🟡 medium: preferences 8.5× over-weighted vs eval.

_GIMP_FN_TO_SKILL = {
    "check_config_status":       "preferences",
    "check_image_size":          "resize",
    "check_image_file_size":     "resize",
    "check_structure_sim":       "structure_only",
    "check_structure_sim_with_threshold": "structure_only",
    "check_structure_sim_resized": "layer_resize",
    "check_palette_and_structure_sim":            "palette",
    "check_brightness_decrease_and_structure_sim": "brightness",
    "check_contrast_increase_and_structure_sim":   "contrast",
    "check_saturation_increase_and_structure_sim": "saturation",
    "check_file_exists_and_structure_sim":         "export_save",
    "check_image_mode_and_structure_sim":          "mode_change",
    "check_color_mode":           "mode_change",
    "check_image_mirror":         "geometry_mirror",
    "check_green_background":     "fill_color_mask",
    "check_textbox_on_leftside":  "canvas_geometry",
    "check_triangle_position":    "canvas_geometry",
    "check_include_exclude":      "shell_eval",
}


def gimp_skill_class(row: dict) -> str:
    f = eval_func_of(row).split("+")[0]
    return _GIMP_FN_TO_SKILL.get(f, f"other:{f}")


# --- vlc: skill_class — by eval_fn → channel ---
# measure_gap docs vlc A 🟡 medium: prefs(vlcrc) under-weighted (-23pp), m3u
# over-weighted (+11pp eval-zero).

_VLC_FN_TO_CHANNEL = {
    "check_qt_bgcone":            "prefs_vlcrc",
    "check_qt_max_volume":        "prefs_vlcrc",
    "check_qt_minimal_view":      "prefs_vlcrc",
    "check_qt_slider_colours":    "prefs_vlcrc",
    "check_global_key_play_pause": "prefs_vlcrc",
    "check_play_and_exit":        "prefs_vlcrc",
    "is_vlc_recordings_folder":   "prefs_vlcrc",
    "check_one_instance_when_started_from_file": "prefs_vlcrc",
    "is_vlc_playing":             "dbus_running",
    "is_vlc_fullscreen":          "dbus_running",
    "compare_images":             "file_image",
    "check_list":                 "file_m3u",
    "compare_audios":             "file_audio",
    "compare_videos":             "file_video",
}


def vlc_skill_class(row: dict) -> str:
    f = eval_func_of(row).split("+")[0]
    return _VLC_FN_TO_CHANNEL.get(f, f"other:{f}")


# --- vs_code: skill_class — by eval_fn → op family ---
# measure_gap docs vs_code 2: structurally aligned, but obscure-key tail under-stressed.

_VS_CODE_FN_TO_SKILL = {
    "check_json_settings":          "settings_json",
    "check_json_keybindings":       "keybindings",
    "compare_text_file":            "file_edit",   # re-bucketed by sub-classifier
    "compare_python_pure_text":     "file_edit",
    "compare_config":               "workspace_config",
    "check_gitignore_has_entries":  "file_create",
    "check_python_file_by_test_suite": "code_quality",
    "diff_text_file":               "file_edit",
    "check_json":                   "settings_json",
}


def vs_code_skill_class(row: dict) -> str:
    """Disambiguates 3 vs_code eval-fn names that v1 conflated:

    - `is_extension_installed`: cmd `code --list-extensions | grep X` (ext_marketplace) vs
      `--install-extension *.vsix` (ext_vsix_local) vs `ls ~/Desktop | grep test.py`
      (file_exists_grep — NOT an extension check at all).
    - `compare_text_file` / `compare_python_pure_text` / `diff_text_file`:
      "create / write / generate" → file_create_template (synth-dominant)
      "indent / replace / append / rename" → file_line_edit (eval-only)
    """
    f = eval_func_of(row).split("+")[0]

    if f == "is_extension_installed":
        cmd = (evaluator_of(row).get("result") or {}).get("command", "")
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        cmd = (cmd or "").lower()
        if "--list-extensions" in cmd or "code --list" in cmd:
            return "ext_marketplace"
        if "vsix" in cmd or "--install-extension" in cmd:
            return "ext_vsix_local"
        if "ls " in cmd and "grep" in cmd:
            return "file_exists_grep"
        return "extensions_other"

    if f in ("compare_text_file", "compare_python_pure_text", "diff_text_file"):
        instr = row["instruction"].lower()
        if any(k in instr for k in ("create a", "write a", "generate a",
                                    "from template", "save it as", "new file")):
            return "file_create_template"
        if any(k in instr for k in ("indent", "replace", "append", "remove",
                                    "edit", "modify", "rename", "tab to space",
                                    "comment out")):
            return "file_line_edit"
        return "file_edit_other"

    return _VS_CODE_FN_TO_SKILL.get(f, f"other:{f}")


# ============================================================================
# Registry
# ============================================================================

Classifier = Callable[[dict], str]

# Common dimensions applied to every domain.
# Style dimensions (path_leak, backtick_any, length_bucket, voice, keystroke,
# menu_arrow) catch generic instruction-style drift that per-domain classifiers
# miss — see the cross-cutting classifier section above.
_COMMON = {
    "atom_count":             atom_count,
    "result_type":            result_type,
    # Loop-1 style leaks
    "path_leak":              path_leak,
    "backtick_any":           backtick_any,
    "length_bucket":          length_bucket,
    "voice":                  voice,
    "keystroke":              keystroke_leak,
    "menu_arrow":             menu_arrow_leak,
    # Loop-2 discourse / config
    "mood":                   mood,
    "sentence_count":         sentence_count_bucket,
    "pre_config_steps":       pre_config_complexity,
    "filename_in_instr":      filename_in_instruction,
    "quoted_value":           quoted_value,
    # Loop-3 discourse / complexity / app-name
    "context_prefix":         context_prefix,
    "oracle_action_complexity": oracle_action_complexity,
    "evaluator_tuning":       evaluator_tuning,
    "app_name_self_ref":      app_name_self_ref,
    # Loop-4 task-essence cross-cutting
    "state_mutation":         state_mutation_class,
    # Loop-5 deeper essence cross-cutting
    "init_file_count":        init_file_count,
    "eval_channel":           eval_channel,
    "oracle_action_sig":      oracle_action_signature,
    # Loop-6 semantic + modality + ambiguity cross-cutting
    "topic_cluster":          topic_cluster,
    "oracle_modality":        oracle_modality,
    "task_ambiguity":         task_ambiguity,
}


# ============================================================================
# Loop-4 task-essence classifiers (per-domain bespoke — each domain looks
# at structural attributes specific to its task family, not the shared style
# axes above). Goal: surface task-DNA differences that drive whether train
# and eval are sampling from "the same distribution" semantically.
# ============================================================================

# --- calc: initial xlsx row-count bucket (synth all <10, eval real-curated 50-500)
def calc_init_row_count(row: dict) -> str:
    for s in _config_steps(row):
        if not isinstance(s, dict): continue
        if s.get("type") == "open":
            return "real_file_open"  # eval pattern — opens a curated xlsx; size unknown here
        params = s.get("parameters", {})
        cmd = params.get("command", "")
        if isinstance(cmd, list): cmd = " ".join(str(x) for x in cmd)
        if not isinstance(cmd, str): continue
        if "openpyxl" not in cmd: continue
        # Try to spot row count from range(...) or list of .append
        m = re.search(r"range\((\d+)\)", cmd)
        n = int(m.group(1)) if m else len(re.findall(r"\.append\s*\(\s*\[", cmd))
        if n == 0: continue
        if n < 10: return "tiny_<10"
        if n < 50: return "small_10-50"
        if n < 200: return "medium_50-200"
        return "large_200+"
    return "unknown"


# --- gimp: image source (synth-PIL vs real-photo) — biggest content gap
def gimp_image_source(row: dict) -> str:
    for s in _config_steps(row):
        if not isinstance(s, dict): continue
        params = s.get("parameters", {})
        cmd = params.get("command", "")
        if isinstance(cmd, list): cmd = " ".join(str(x) for x in cmd)
        if not isinstance(cmd, str): cmd = ""
        if "Image.new" in cmd or "from PIL" in cmd:
            return "synth_pil_pattern"
        if "hf_hub_download" in cmd or "huggingface" in cmd.lower():
            return "real_photo"
        if s.get("type") in ("download", "host_push"):
            return "real_photo"
    return "unknown_source"


# --- gimp: operation class (transform vs filter vs color_adjust vs layer)
_GIMP_OP_PATTERNS = [
    ("transform_geom", re.compile(r"\b(rotate|flip|mirror|transpose|rotat)", re.I)),
    ("transform_size", re.compile(r"\b(crop|resize|scale|shrink|enlarge)", re.I)),
    ("filter_apply",   re.compile(r"\b(blur|sharpen|emboss|vignette|pixel|posteriz|noise|filter|distort)", re.I)),
    ("color_adjust",   re.compile(r"\b(brightness|contrast|saturation|hue|level|curve|temperature|colorize|grayscal|invert|exposure)", re.I)),
    ("layer_op",       re.compile(r"\b(layer|mask|opacity|flatten|merge)", re.I)),
    ("text_op",        re.compile(r"\b(text|caption|label|annotat|stamp)", re.I)),
    ("export_op",      re.compile(r"\b(export|save as|convert)", re.I)),
    ("config_op",      re.compile(r"\b(theme|preference|setting|configur)", re.I)),
    ("background",     re.compile(r"\b(background|transparent|alpha)", re.I)),
]
def gimp_op_class(row: dict) -> str:
    instr = row["instruction"]
    for name, pat in _GIMP_OP_PATTERNS:
        if pat.search(instr): return name
    return "other"


# --- os: persistence target (more granular than system_target)
def os_persistence_target(row: dict) -> str:
    e = evaluator_of(row)
    res = e.get("result", {})
    cmd = ""
    if isinstance(res, dict):
        c = res.get("command", "")
        cmd = " ".join(str(x) for x in c) if isinstance(c, list) else str(c)
    elif isinstance(res, list):
        for r2 in res:
            if isinstance(r2, dict):
                c = r2.get("command", "")
                cmd += " " + (" ".join(str(x) for x in c) if isinstance(c, list) else str(c))
    cmd_l = cmd.lower()
    if "gsettings" in cmd_l:     return "gsettings"
    if "dconf" in cmd_l:         return "dconf"
    if "systemctl" in cmd_l or "/etc/systemd/" in cmd_l: return "systemd"
    if "crontab" in cmd_l or "/etc/cron" in cmd_l: return "cron"
    if "pactl" in cmd_l or "xfconf" in cmd_l or "timedatectl" in cmd_l: return "sys_daemon"
    if "/etc/" in cmd_l:         return "etc_file"
    if "/home/user" in cmd_l or "$home" in cmd_l: return "home_file"
    if "bash" in cmd_l and "eval.sh" in cmd_l: return "shell_eval_script"
    return "other"


# --- thunderbird: initial inbox/folder state
def thunderbird_init_state(row: dict) -> str:
    cmds = ""
    for s in _config_steps(row):
        if not isinstance(s, dict): continue
        p = s.get("parameters", {})
        c = p.get("command", "")
        cmds += " " + (" ".join(str(x) for x in c) if isinstance(c, list) else str(c))
    cmds_l = cmds.lower()
    if re.search(r"\.(eml|emlx|mbox)\b|gloda\.sqlite", cmds_l): return "populated_inbox"
    if "msgfilterrules" in cmds_l: return "pre_filters"
    if re.search(r"(local folders|imap|mail/)", cmds_l): return "folder_setup_only"
    return "minimal"


# --- multi_apps: state-handoff mechanism between apps
_HANDOFF_CLIP = re.compile(r"\b(copy|paste|clipboard)\b", re.I)
_HANDOFF_WRITE = re.compile(r"\b(save|export|write|store|output)\b", re.I)
_HANDOFF_READ  = re.compile(r"\b(open|read|load|import|attach|cite|reference)\b", re.I)
def multi_apps_handoff(row: dict) -> str:
    instr = row["instruction"]
    has_clip = bool(_HANDOFF_CLIP.search(instr))
    has_w = bool(_HANDOFF_WRITE.search(instr))
    has_r = bool(_HANDOFF_READ.search(instr))
    if has_clip:      return "clipboard"
    if has_w and has_r: return "file_io_bidirectional"
    if has_w:         return "write_only"
    if has_r:         return "read_only"
    return "other_handoff"


# --- chrome: evaluator inspection depth
def chrome_inspection_depth(row: dict) -> str:
    res = evaluator_of(row).get("result", {})
    if not isinstance(res, dict): return "other"
    typ = res.get("type", "")
    if typ in ("active_tab_url", "active_tab_url_parse", "url_path_parse", "url_dashPart"):
        return "url_shallow"
    if typ in ("active_tab_html_parse", "active_tab_info", "page_info"):
        return "page_html_deep"
    if typ in ("active_url_from_accessTree", "accessibility_tree"):
        return "accessibility_tree"
    if typ in ("bookmarks", "open_tabs_info", "history", "cookies"):
        return "browser_internals"
    if typ in ("shortcuts_on_desktop", "default_search_engine", "profile_name",
               "enable_do_not_track", "enable_safe_browsing", "chrome_color_scheme",
               "chrome_font_size", "new_startup_page", "data_delete_automacally"):
        return "settings_state"
    return "other"


# --- vs_code: workspace shape
def vs_code_workspace_shape(row: dict) -> str:
    n_files = 0
    has_workspace_file = False
    for s in _config_steps(row):
        if not isinstance(s, dict): continue
        p = s.get("parameters", {})
        cmd = p.get("command", "")
        if isinstance(cmd, list): cmd = " ".join(str(x) for x in cmd)
        if not isinstance(cmd, str): continue
        if ".code-workspace" in cmd: has_workspace_file = True
        n_files += len(re.findall(r"open\s*\(['\"](?:/home/user|~)/", cmd))
    if has_workspace_file: return "workspace_file"
    if n_files >= 3:       return "multi_file_dir"
    if n_files >= 1:       return "single_file"
    return "no_init_files"


# --- writer: content provenance (real vs Lorem)
def writer_content_provenance(row: dict) -> str:
    """Real-content (Gutenberg/Wikipedia/HF cache) vs synth-inline lorem."""
    has_download = False
    has_inline_text = False
    for s in _config_steps(row):
        if not isinstance(s, dict): continue
        if s.get("type") in ("download", "host_push"):
            has_download = True
            continue
        p = s.get("parameters", {})
        cmd = p.get("command", "")
        if isinstance(cmd, list): cmd = " ".join(str(x) for x in cmd)
        if isinstance(cmd, str) and ("from docx" in cmd or "Document()" in cmd or "add_paragraph" in cmd):
            has_inline_text = True
    if has_download:    return "real_corpus_docx"
    if has_inline_text: return "synth_inline_docx"
    return "other_source"


# --- impress: target-slide scope (slide-N anchor vs deck-wide ops)
def impress_target_scope(row: dict) -> str:
    instr = row["instruction"]
    if re.search(r"\b(every slide|all slides|across the deck|throughout|whole presentation|entire deck|each slide)\b", instr, re.I):
        return "deck_wide"
    if re.search(r"\bslides?\s+\d+\s*[,&]\s*\d+", instr):
        return "multi_slide_list"
    if re.search(r"\bslide\s+\d+\b", instr, re.I):
        return "single_slide_ordinal"
    if re.search(r"\bfirst|second|third|last|next|previous\s+slide", instr, re.I):
        return "relative_slide"
    return "deck_wide_or_implicit"


DOMAIN_DIMENSIONS: dict[str, dict[str, Classifier]] = {
    "os": {
        "instruction_style":  os_instruction_style,
        "skill_scope":        os_skill_scope,
        "system_target":      os_system_target,
        "difficulty_nl2bash": os_difficulty_nl2bash,
        "persistence_target": os_persistence_target,
        **_COMMON,
    },
    "chrome": {
        "eval_fn_family":    chrome_eval_fn_family,
        "url_leak":          chrome_url_leak,
        "slot_resolution":   chrome_slot_resolution,
        "relative_time":     chrome_relative_time,
        "inspection_depth":  chrome_inspection_depth,
        **_COMMON,
    },
    "libreoffice_writer": {
        "skill_class":        writer_skill_class,
        "target_anchor":      writer_target_anchor,
        "evaluator_pattern":  writer_evaluator_pattern,
        "content_provenance": writer_content_provenance,
        **_COMMON,
    },
    "multi_apps": {
        "apps_per_task":    multi_apps_per_task,
        "app_combination":  multi_apps_combination,
        "tool_leak":        multi_apps_tool_leak,
        "handoff":          multi_apps_handoff,
        "eval_fn":          eval_func_of,
        **_COMMON,
    },
    "libreoffice_calc": {
        "save_protocol":      calc_save_protocol,
        "skill_class":        calc_skill_class,
        "rule_combo":         calc_rule_combo,
        "source_provenance":  calc_source_provenance,
        "init_row_count":     calc_init_row_count,
        "eval_fn":            eval_func_of,
        **_COMMON,
    },
    "libreoffice_impress": {
        "comparator_strictness": impress_comparator,
        "op_family":             impress_op_family,
        "rgb_leak":              impress_rgb_leak,
        "slide_anchor":          impress_slide_anchor,
        "target_scope":          impress_target_scope,
        "eval_fn":               eval_func_of,
        **_COMMON,
    },
    "gimp": {
        "instruction_leak": gimp_instruction_leak,
        "skill_class":      gimp_skill_class,
        "image_source":     gimp_image_source,
        "op_class":         gimp_op_class,
        "eval_fn":          eval_func_of,
        **_COMMON,
    },
    "thunderbird": {
        "instruction_leak": thunderbird_instruction_leak,
        "async_flush":      thunderbird_async_flush,
        "pref_key_family":  thunderbird_pref_key_family,
        "init_state":       thunderbird_init_state,
        "eval_fn":          eval_func_of,
        **_COMMON,
    },
    "vlc": {
        "instruction_leak": vlc_instruction_leak,
        "skill_class":      vlc_skill_class,
        "media_source":     vlc_media_source,
        "eval_fn":          eval_func_of,
        **_COMMON,
    },
    "vs_code": {
        "instruction_leak": vs_code_instruction_leak,
        "skill_class":      vs_code_skill_class,
        "workspace_shape":  vs_code_workspace_shape,
        "eval_fn":          eval_func_of,
        **_COMMON,
    },
}


# ============================================================================
# Manual calibration targets from /measure_gap docs
# Format: MANUAL_TARGETS[domain][dim][category] = {"synth": pct, "eval": pct}
# Either field may be omitted if the manual measure_gap docs didn't pin a number.
# Quant within ±5pp of manual → ✓cal; else ❌cal.
# ============================================================================

MANUAL_TARGETS: dict[str, dict[str, dict[str, dict[str, float]]]] = {
    "os": {
        # NOTE: measure_gap docs OS section was written before cycle-46 rebalance
        # (#136 added 6 GUI-Settings templates). Targets below reflect
        # **post-rebalance** reality — the bridge fix has partially landed.
        "instruction_style": {
            "backtick_leak": {"synth": 88.0, "eval": 4.0},
            "user_voice":    {"synth": 12.0, "eval": 96.0},
        },
        "skill_scope": {
            # Eval percentages are post-infeasibility-filter (N=19 not 24).
            # synth has ~18 sys-state rows after cycle-46 rebalance.
            "gui_settings":   {"synth": 22.0, "eval": 37.0},
            "shell_pipeline": {"synth": 21.0, "eval": 21.0},
            "file_edit":      {"synth": 57.0, "eval": 42.0},
        },
        "system_target": {
            "userspace_desktop": {"synth": 55.0, "eval": 11.0},
            "userspace_dotfile": {"synth": 18.0, "eval": 0.0},
            "gsettings_dconf":   {"synth": 16.0, "eval": 21.0},
            "sys_daemon":        {"synth": 6.0,  "eval": 16.0},
        },
        "difficulty_nl2bash": {
            "multistep_bash": {"eval": 25.0},
        },
    },
    "chrome": {
        "url_leak": {
            "url_leaked":   {"eval": 0.0},
            "url_implicit": {"eval": 100.0},
        },
        # Quant correction: eval also pre-resolves ~79% URLs in config (not 0% as measure_gap docs framed).
        # Real synth/eval gap on this axis is small.
        "slot_resolution": {
            "agent_must_navigate":    {"synth": 27.0, "eval": 21.0},
            "config_preresolved_url": {"synth": 73.0, "eval": 79.0},
        },
        # Eval has 5 rule_relativeTime rows; synth has 0. J-archetype gap.
        "relative_time": {
            "relative_time": {"synth": 0.0, "eval": 12.0},
        },
    },
    "libreoffice_writer": {
        # measure_gap docs writer (corrected examine_* decomposition).
        "skill_class": {
            "tables":            {"synth": 1.5,  "eval": 13.0},   # 🔴 under (headline)
            "specialized_uncov": {"synth": 10.7, "eval": 26.1},   # 🔴 residual gap after partial fix
            "pdfs":              {"synth": 2.0,  "eval": 4.3},
            "font_name":         {"synth": 15.8, "eval": 8.7},
            "images":            {"synth": 10.2, "eval": 4.3},
            "line_spacing":      {"synth": 9.7,  "eval": 8.7},
            "text_match":        {"synth": 39.3, "eval": 21.7},
        },
        "target_anchor": {
            "quote_anchor": {"synth": 0.0, "eval": 9.0},
            "ordinal":      {"synth": 37.0, "eval": 9.0},
        },
        # Surfaces the synth dual-evaluator pattern and the 5-row eval compound
        # hole. eval never uses compare_docx_strict; synth uses it 34% of time.
        "evaluator_pattern": {
            "compare_docx_strict+examine_flag": {"synth": 18.0, "eval": 0.0},
            "compare_docx_strict_default":      {"synth": 16.0, "eval": 0.0},
            "specific_upstream_fn":             {"synth": 66.0, "eval": 77.0},
            "compound_multi_property":          {"synth": 0.0,  "eval": 23.0},
        },
    },
    "multi_apps": {
        # measure_gap docs multi_apps claimed 84% synth single-app vs 2% eval. With
        # combined-signal detection (instruction kw ∪ eval-fn inference) the
        # gap shrinks to 61% vs 53% — the manual claim was instruction-only
        # and over-stated. Real gap is small; check the app_combination
        # heatmap for missing pair shapes (thunderbird+*, gimp, pdf+writer).
        # After eval-fn-inference expansion (gdrive/git/gnome/vim apps + 4 new
        # eval-fn mappings) eval-side apps_le_1 dropped 52% → 46%, apps_3plus
        # rose 5% → 8%. Synth still 0 apps_3plus — coverage hole.
        "apps_per_task": {
            "apps_le_1":  {"synth": 61.0, "eval": 46.0},
            "apps_2":     {"synth": 39.0, "eval": 46.0},
            "apps_3plus": {"synth": 0.0,  "eval": 8.0},
        },
        # measure_gap docs multi_apps F: synth leaks pdftk/pandoc/IM exact commands.
        "tool_leak": {
            "tool_leak": {"eval": 0.0},   # eval names the tool, not the flags
        },
    },
    "libreoffice_calc": {
        "save_protocol": {
            "ctrl_s_only":  {"synth": 100.0, "eval": 0.0},
            "open+ctrl_s":  {"synth": 0.0,   "eval": 94.0},
        },
        # measure_gap docs calc C 🔴 large: synth 0% pivot/sheet_print; eval has both.
        "skill_class": {
            "pivot_table": {"synth": 0.0, "eval": 11.0},
            "sheet_print": {"synth": 0.0, "eval": 11.0},
            "check_cell":  {"synth": 0.0, "eval": 4.0},
        },
        # Synth=98% inline-openpyxl vs eval=100% curated download .xlsx.
        # Biggest content gap; invisible at fn-name level.
        "source_provenance": {
            "synth_inline_openpyxl": {"synth": 98.0, "eval": 0.0},
            "curated_real_xlsx":     {"synth": 0.0,  "eval": 100.0},
        },
    },
    "libreoffice_impress": {
        "comparator_strictness": {
            "color_tolerant":    {"synth": 29.0, "eval": 0.0},
            "position_tolerant": {"synth": 3.0,  "eval": 0.0},
        },
        # measure_gap docs said eval 23%; quant shows 36% — measure_gap docs was instr-only undercount.
        "op_family": {
            "title_or_body_style": {"synth": 52.0, "eval": 36.0},
        },
        "rgb_leak": {
            "rgb_triplet_leak": {"eval": 0.0},
        },
        # Eval has 12 multi-fn compound rows (per-slide diff). Synth: 0.
        "atom_count": {
            "atom_2":     {"synth": 0.0, "eval": 26.0},
            "atom_3plus": {"synth": 0.0, "eval": 0.0},
        },
        # Quant: measure_gap docs' "10% eval" was an instruction-skim estimate; my regex
        # catches only the strict `'X' slide` shape. Real gap is small (0.4/2.1%).
        # The deeper title-anchor signal lives in eval's *.pptx file content,
        # not the instruction — left for v3.
        "slide_anchor": {
            "title_text_anchor": {"synth": 0.0, "eval": 2.0},
        },
    },
    "gimp": {
        # measure_gap docs gimp A: 82/100 instructions leak Ctrl+L / menu paths.
        "instruction_leak": {
            "leak":    {"synth": 82.0, "eval": 0.0},
        },
        # Quant: after filtering 10 eval infeasibilities (mostly preferences-shaped),
        # eval-side preferences rate jumps 4% → 25%. Gap is 9pp not 30pp; still ⚠️.
        "skill_class": {
            "preferences": {"synth": 34.0, "eval": 25.0},
        },
    },
    "thunderbird": {
        "instruction_leak": {
            "pref_key_leak": {"eval": 0.0},
        },
        # POLARITY-CORRECTED v2 (measure_gap docs was inverted): eval canonical is
        # `close_window_only` (57%) + some `no_postconfig` rows; synth invented
        # `pkill_kill_signal` (30%) — this IS the gap (over-aggressive shutdown).
        "async_flush": {
            "pkill_kill_signal":  {"synth": 30.0, "eval": 0.0},
            "close_window_only":  {"synth": 64.0, "eval": 57.0},
        },
    },
    "vlc": {
        "instruction_leak": {
            "menu_path_leak": {"eval": 0.0},
        },
        # Post-infeasibility-filter eval prefs_vlcrc share rises 47% → 53%.
        "skill_class": {
            "prefs_vlcrc": {"synth": 24.0, "eval": 53.0},
            "file_m3u":    {"synth": 11.0, "eval": 0.0},
        },
        # measure_gap docs vlc B: bba3381f HLS streaming 0 synth coverage.
        "media_source": {
            "hls_m3u8":         {"synth": 0.0, "eval": 7.0},
            "remote_url_media": {"synth": 0.0},
        },
    },
    "vs_code": {
        # measure_gap docs vs_code 4: backticked JSON keys in ~30 templates.
        "instruction_leak": {
            "key_leak": {"eval": 0.0},
        },
        # Quant: post-infeasibility-filter (5/23 excluded), settings_json
        # rises 30% → 39% eval. With sub-classifier splitting extensions into
        # ext_marketplace/ext_vsix_local/file_exists_grep, the old "extensions"
        # bucket no longer appears — replaced with finer-grained shares.
        "skill_class": {
            "settings_json":   {"synth": 39.0, "eval": 39.0},
            "ext_marketplace": {"eval": 17.0},   # 3/18 after filter
        },
    },
}


# ============================================================================
# Compute + print
# ============================================================================

def compute_distribution(rows: list[dict], classifier: Classifier) -> tuple[dict[str, tuple[int, float]], int]:
    counts: Counter[str] = Counter()
    for r in rows:
        counts[classifier(r)] += 1
    n = len(rows)
    return {k: (v, (v / n * 100 if n else 0.0)) for k, v in counts.items()}, n


def status_flag(synth_pct: float, eval_pct: float) -> str:
    if eval_pct > 0 and synth_pct == 0:
        return "❌"
    delta = abs(synth_pct - eval_pct)
    if delta < 5: return "✓"
    if delta < 15: return "⚠️"
    return "🔴"


def cal_flag(domain: str, dim: str, category: str, synth_pct: float, eval_pct: float) -> str:
    target = MANUAL_TARGETS.get(domain, {}).get(dim, {}).get(category)
    if target is None:
        return "    "  # no manual target
    deltas = []
    if "synth" in target:
        deltas.append(abs(synth_pct - target["synth"]))
    if "eval" in target:
        deltas.append(abs(eval_pct - target["eval"]))
    if not deltas:
        return "    "
    return "✓cal" if max(deltas) <= 5.0 else "❌cal"


def print_dimension(domain: str, dim_name: str,
                    synth_dist: dict, synth_n: int,
                    eval_dist: dict, eval_n: int,
                    only_calibration: bool = False,
                    min_count: int = 0) -> None:
    cats = sorted(set(synth_dist) | set(eval_dist),
                  key=lambda c: -max(synth_dist.get(c, (0, 0))[1],
                                     eval_dist.get(c, (0, 0))[1]))
    if only_calibration:
        cats = [c for c in cats
                if MANUAL_TARGETS.get(domain, {}).get(dim_name, {}).get(c)]
        if not cats:
            return
    if min_count > 0:
        cats = [c for c in cats
                if max(synth_dist.get(c, (0, 0))[0],
                       eval_dist.get(c, (0, 0))[0]) >= min_count]
    print(f"  -- {dim_name}  (synth N={synth_n}, eval N={eval_n}) --")
    print(f"    {'category':<26} {'synth_n':>7} {'synth%':>7} {'eval_n':>7} {'eval%':>7} {'Δpp':>7}  {'status':<6} cal")
    for c in cats:
        sn, sp = synth_dist.get(c, (0, 0.0))
        en, ep = eval_dist.get(c, (0, 0.0))
        dpp = sp - ep
        sym = status_flag(sp, ep)
        cal = cal_flag(domain, dim_name, c, sp, ep)
        print(f"    {c:<26} {sn:>7d} {sp:>6.1f}% {en:>7d} {ep:>6.1f}% {dpp:>+6.1f}  {sym:<6} {cal}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", nargs="*", default=None,
                    help="domain(s) to analyze; default = all registered")
    ap.add_argument("--calibration", action="store_true",
                    help="only show categories with manual targets (calibration view)")
    ap.add_argument("--min-count", type=int, default=0,
                    help="suppress categories with max(synth_n, eval_n) below this")
    ap.add_argument("--synth-path", type=Path, default=SYNTH_PATH,
                    help="override synth jsonl path (for sandboxed subagent runs)")
    args = ap.parse_args()

    synth_path = args.synth_path
    try:
        synth_display = synth_path.relative_to(REPO_ROOT)
    except ValueError:
        synth_display = synth_path
    print(f"Loading {synth_display} ...")
    synth_all = load_jsonl(synth_path)
    print(f"Loading {EVAL_PATH.relative_to(REPO_ROOT)} ...")
    eval_all = load_jsonl(EVAL_PATH)

    # Filter infeasibility — train never includes (AGENTS.md INFEASIBLE_CLAIM_TRAIN).
    # Including them in gap dimensions would conflate "by-design eval-only signal"
    # with "synth coverage hole".
    eval_infeas = sum(1 for r in eval_all if is_infeasibility(r))
    synth_infeas = sum(1 for r in synth_all if is_infeasibility(r))
    eval_all = [r for r in eval_all if not is_infeasibility(r)]
    synth_all = [r for r in synth_all if not is_infeasibility(r)]
    print(f"Synth N={len(synth_all)} (filtered {synth_infeas} infeasibility — "
          f"must be 0 per AGENTS.md), Eval N={len(eval_all)} "
          f"(filtered {eval_infeas} infeasibility — eval-only by design)\n")

    if synth_infeas > 0:
        print(f"⚠️  WARNING: {synth_infeas} synth row(s) have infeasibility "
              f"evaluator — this violates AGENTS.md INFEASIBLE_CLAIM_TRAIN. "
              f"These rows should be removed from train.synth.jsonl.\n")

    target_domains = args.domain or list(DOMAIN_DIMENSIONS.keys())

    # Macro summary — density imbalance, instruction length disparity.
    # These are absolute / mean-based metrics that the per-domain Δpp tables
    # cannot surface (a domain can be aligned in every category yet still be
    # 10× oversampled in row count, or 4× longer in mean instruction length).
    _print_density_and_length_summary(synth_all, eval_all, target_domains)

    for domain in target_domains:
        if domain not in DOMAIN_DIMENSIONS:
            print(f"!! domain={domain!r} not in registry; "
                  f"registered: {list(DOMAIN_DIMENSIONS)}")
            continue
        synth_rows = [r for r in synth_all if domain_of(r) == domain]
        eval_rows = [r for r in eval_all if domain_of(r) == domain]
        print(f"=== {domain} (synth={len(synth_rows)}, eval={len(eval_rows)}) ===")
        for dim, fn in DOMAIN_DIMENSIONS[domain].items():
            synth_dist, sn = compute_distribution(synth_rows, fn)
            eval_dist, en = compute_distribution(eval_rows, fn)
            print_dimension(domain, dim, synth_dist, sn, eval_dist, en,
                            only_calibration=args.calibration,
                            min_count=args.min_count)
        print()

    # Micro summary — per-eval-task coverage holes.
    # For each eval task, find the closest same-domain synth row by (a) eval_fn
    # equality (preferred) and (b) instruction token Jaccard overlap (fallback).
    # Eval tasks with no same-fn synth analog AND/OR very low token-overlap to
    # best fuzzy match are the long-tail coverage holes that the categorical
    # Δpp tables average out.
    if not args.calibration:
        _print_coverage_holes(synth_all, eval_all, target_domains)


def _distinct_n_ratio(rows: list[dict], n: int = 2) -> float:
    """Distinct-N: ratio of unique n-grams to total n-grams across all
    instructions in the row set. Higher = more lexically diverse.

    A synth set that is heavy on a few templates will have low distinct-N
    relative to a curated eval set, even if individual rows look "different"
    superficially. Catches paraphrase-flood subagent edits.
    """
    import re as _re
    grams_total = 0
    grams_unique: set[tuple[str, ...]] = set()
    for r in rows:
        toks = _re.findall(r"[a-zA-Z']+", r["instruction"].lower())
        if len(toks) < n: continue
        for i in range(len(toks) - n + 1):
            grams_unique.add(tuple(toks[i:i+n]))
            grams_total += 1
    return len(grams_unique) / grams_total if grams_total else 0.0


def _uniq_instruction_ratio(rows: list[dict]) -> float:
    """Fraction of rows whose normalized instruction is unique within the set.
    Detects literal template-cloning (instruction copy-pasted across many rows
    with only slot fills differing). Eval is 100% unique by curation; healthy
    synth should be ≥95%."""
    import re as _re
    def _norm(s: str) -> str:
        # collapse digits + paths + quoted slots so slot-fills don't inflate uniqueness
        s = _re.sub(r"\d+", "#", s.lower())
        s = _re.sub(r"/(?:home|tmp|etc|var|opt)/\S+", "/PATH", s)
        s = _re.sub(r"['\"][^'\"]{1,40}['\"]", "QSTR", s)
        s = _re.sub(r"\s+", " ", s).strip()
        return s
    if not rows: return 0.0
    norms = [_norm(r["instruction"]) for r in rows]
    return len(set(norms)) / len(norms)


def _template_clone_factor(rows: list[dict]) -> float:
    """Avg rows per template-base (task_id with _NNNN suffix stripped).
    Synth often emits 2 paraphrased rows per base (Cap-2 design); eval = 1.0
    by curation. Higher synth value indicates more template-cloning."""
    import re as _re
    if not rows: return 0.0
    bases = set(_re.sub(r"_\d{4}$", "", r.get("task_id", "")) for r in rows)
    return len(rows) / len(bases) if bases else 0.0


def _print_density_and_length_summary(
    synth_all: list[dict], eval_all: list[dict], domains: list[str]
) -> None:
    """Macro report covering metrics that per-domain Δpp tables can't surface:
    - Density: synth/eval row-count ratio (oversample / undersample detection)
    - Domain mix: synth's % share vs eval's % share per domain (the model
      training-time prior on which domain a task belongs to)
    - Instruction length: mean chars; flags >50% drift
    - Instruction diversity: distinct-2 ratio (lower = more templated);
      unique-norm-instruction ratio (lower = literal template-clones)
    """
    syn_by_dom: dict[str, list[dict]] = {}
    ev_by_dom: dict[str, list[dict]] = {}
    for r in synth_all:
        syn_by_dom.setdefault(domain_of(r), []).append(r)
    for r in eval_all:
        ev_by_dom.setdefault(domain_of(r), []).append(r)

    ratios = []
    for d in domains:
        sn, en = len(syn_by_dom.get(d, [])), len(ev_by_dom.get(d, []))
        if en: ratios.append(sn / en)
    if not ratios: return
    ratios.sort()
    median_ratio = ratios[len(ratios) // 2]
    syn_total = sum(len(syn_by_dom.get(d, [])) for d in domains)
    ev_total = sum(len(ev_by_dom.get(d, [])) for d in domains)

    print("=== macro: synth↔eval density / length / diversity / mix / clone-factor ===")
    print(f"{'domain':<22} {'sN':>5} {'eN':>4} {'ratio':>6} "
          f"{'s%mix':>5} {'e%mix':>5} {'Δmix':>6}  "
          f"{'sLen':>5} {'eLen':>5} {'Δlen':>5}  "
          f"{'sD2':>5} {'eD2':>5} {'Δd2':>5}  "
          f"{'sUniq':>5} {'eUniq':>5}  "
          f"{'sClone':>6}  status")
    flag_summary: list[str] = []
    for d in domains:
        sr = syn_by_dom.get(d, [])
        er = ev_by_dom.get(d, [])
        sn, en = len(sr), len(er)
        if not en:
            print(f"{d:<22} {sn:>5} {en:>4} {'—':>6}")
            continue
        ratio = sn / en
        s_mix = 100 * sn / syn_total if syn_total else 0
        e_mix = 100 * en / ev_total if ev_total else 0
        dmix = s_mix - e_mix
        slen = sum(len(r["instruction"]) for r in sr) / max(sn, 1)
        elen = sum(len(r["instruction"]) for r in er) / max(en, 1)
        dlen = slen - elen
        s_d2 = _distinct_n_ratio(sr, 2)
        e_d2 = _distinct_n_ratio(er, 2)
        d_d2 = s_d2 - e_d2
        s_uq = _uniq_instruction_ratio(sr)
        e_uq = _uniq_instruction_ratio(er)
        s_clone = _template_clone_factor(sr)
        flags = []
        if ratio >= 2 * median_ratio:    flags.append("oversampled")
        elif ratio <= 0.5 * median_ratio: flags.append("undersampled")
        if abs(dmix) >= 8:               flags.append("mix-drift")
        if elen and abs(dlen)/elen > 0.5: flags.append("len-drift")
        if e_d2 and s_d2 < 0.7 * e_d2:   flags.append("templated")
        if s_uq < 0.85:                  flags.append("clone-rows")
        if s_clone >= 1.8:               flags.append("heavy-clone-factor")
        sym = "🔴" if flags else "✓"
        print(f"{d:<22} {sn:>5} {en:>4} {ratio:>5.1f}x "
              f"{s_mix:>4.1f}% {e_mix:>4.1f}% {dmix:>+5.1f}  "
              f"{slen:>5.0f} {elen:>5.0f} {dlen:>+5.0f}  "
              f"{s_d2:>4.2f} {e_d2:>4.2f} {d_d2:>+5.2f}  "
              f"{s_uq:>4.2f} {e_uq:>4.2f}  "
              f"{s_clone:>5.2f}  {sym} {' '.join(flags)}")
        if flags: flag_summary.append(f"  - {d}: {', '.join(flags)}")
    if flag_summary:
        print("\nMacro flags:")
        for s in flag_summary: print(s)
    print()


def _print_coverage_holes(
    synth_all: list[dict], eval_all: list[dict], domains: list[str]
) -> None:
    """Per-domain report of eval tasks with no same-fn synth analog, and the
    compound-evaluator signatures that have zero synth coverage."""
    import re as _re

    def _tokens(s: str) -> set[str]:
        return set(_re.findall(r"[a-z]{4,}", s.lower()))

    syn_by_dom: dict[str, list[dict]] = {}
    ev_by_dom: dict[str, list[dict]] = {}
    for r in synth_all: syn_by_dom.setdefault(domain_of(r), []).append(r)
    for r in eval_all: ev_by_dom.setdefault(domain_of(r), []).append(r)

    print("=== micro: per-eval-task coverage holes ===\n")

    # 1. Per-domain count of eval tasks with no same-fn synth.
    total_no_fn = 0
    print(f"{'domain':<22} {'no_fn':>5} {'weak_match (<.15 Jaccard)':>26}")
    breakdown: dict[str, dict] = {}
    for d in domains:
        srows = syn_by_dom.get(d, [])
        erows = ev_by_dom.get(d, [])
        no_fn = []
        weak = []
        for er in erows:
            ef = eval_func_of(er)
            same_fn = [s for s in srows if eval_func_of(s) == ef]
            if not same_fn:
                no_fn.append(er)
                continue
            et = _tokens(er["instruction"])
            best_sim = max(
                len(et & _tokens(s["instruction"])) / max(len(et | _tokens(s["instruction"])), 1)
                for s in same_fn
            )
            if best_sim < 0.15:
                weak.append((best_sim, er))
        breakdown[d] = {"no_fn": no_fn, "weak": weak}
        total_no_fn += len(no_fn)
        print(f"{d:<22} {len(no_fn):>5} {len(weak):>26}")
    print(f"\nTotal eval tasks with no same-fn synth analog: {total_no_fn}/{len(eval_all)}")

    # 2. Per-domain: list the eval task_ids of no-fn holes (so the data-curator
    # can grep). Cap at 10 per domain for readability.
    print("\n--- sample no-fn-analog eval tasks (≤10 per domain) ---")
    for d in domains:
        nofn = breakdown.get(d, {}).get("no_fn", [])
        if not nofn: continue
        print(f"\n[{d}] {len(nofn)} task(s):")
        for er in nofn[:10]:
            print(f"  {er['task_id']:<60} {eval_func_of(er)[:40]:<40} :: "
                  f"{er['instruction'][:80]}")
        if len(nofn) > 10:
            print(f"  ... +{len(nofn)-10} more")

    # 3. Compound-evaluator signatures with zero synth coverage.
    print("\n--- compound-evaluator signatures with ZERO synth coverage ---")
    from collections import Counter as _Counter
    for d in domains:
        srows = syn_by_dom.get(d, [])
        erows = ev_by_dom.get(d, [])
        syn_fns = _Counter(eval_func_of(r) for r in srows)
        compounds_unmet = []
        for er in erows:
            ef = eval_func_of(er)
            if "+" in ef and syn_fns.get(ef, 0) == 0:
                compounds_unmet.append(ef)
        if not compounds_unmet: continue
        c = _Counter(compounds_unmet)
        print(f"[{d}]")
        for sig, n in c.most_common():
            print(f"  n={n:>2}  {sig}")


if __name__ == "__main__":
    main()
