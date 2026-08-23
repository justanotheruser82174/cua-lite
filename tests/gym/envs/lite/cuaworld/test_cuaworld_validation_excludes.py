"""CUAWorld tests split from _cuaworld_support.py: validation excludes."""
from __future__ import annotations

import json

import pytest

from lite.gym.envs.lite.cuaworld.src import software
from lite.gym.errors import EnvDepsMissingError
from tests.gym.envs.lite._cuaworld_support import _cuaworld_root, _materials_root


def test_validation_excludes_total_matches_its_entries():
    """`_meta._total` is the file's own checksum; a hand-edit that forgets it makes
    every downstream count wrong."""
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )
    entries = sum(
        len(tasks)
        for name, tasks in excludes.items()
        if not software.is_excludes_metadata_key(name)
    )
    assert entries == excludes["_meta"]["_total"]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "{",
        "[]",
        json.dumps({"_meta": {"_total": 2}, "gcompris": {"task": "other"}}),
        json.dumps({"_meta": {"_total": 1}, "gcompris": {"task": ""}}),
    ],
)
def test_validation_excludes_loader_fails_closed(tmp_path, monkeypatch, payload):
    path = tmp_path / "validation_excludes.json"
    if payload is not None:
        path.write_text(payload)
    monkeypatch.setattr(software, "VALIDATION_EXCLUDES_PATH", path)
    monkeypatch.setattr(software, "_EXCLUDE_REASONS", None)

    with pytest.raises(EnvDepsMissingError, match="validation_excludes.json"):
        software._exclude_reasons()


def test_validation_excludes_loader_validates_and_strips_metadata(tmp_path, monkeypatch):
    path = tmp_path / "validation_excludes.json"
    path.write_text(
        json.dumps({
            "_meta": {"_total": 1},
            "_note": {"comment": "metadata keys are ignored"},
            "gcompris": {"share_candies_division": "other"},
        })
    )
    monkeypatch.setattr(software, "VALIDATION_EXCLUDES_PATH", path)
    monkeypatch.setattr(software, "_EXCLUDE_REASONS", None)

    assert software._exclude_reasons() == {
        "gcompris": {"share_candies_division": "other"}
    }


def test_b22_live_rollout_gcompris_blank_activity_is_excluded():
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )
    assert excludes["gcompris"]["share_candies_division"] == "other"


def test_openvsp_wave_drag_export_nameerror_is_excluded():
    """`openvsp_wave_drag_area_ruling/export_result.sh:89` interpolates a bare
    `{RESULT_FILE}` into an f-string. `RESULT_FILE` is a SHELL variable (set at :7),
    the heredoc is unquoted so `${RESULT_FILE}` WOULD have expanded, but `{…}` does
    not — python raises NameError before the `mv`, so `/tmp/wave_drag_result.json`
    never exists and the verifier's `copy_from_env` reports "Result JSON not found"
    on every run. The `_HOOK_SITECUSTOMIZE` shim cannot help: it binds only
    true/false/null. A sibling task (`openvsp_external_stores_integration`) uses the
    correct `'${RESULT_FILE}'` spelling and must not be touched.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )
    assert excludes["openvsp"]["openvsp_wave_drag_area_ruling"] == "verifier_nameerror"

    export = next(
        iter(
            _materials_root().glob(
                "openvsp/*/tasks/openvsp_wave_drag_area_ruling/"
                "export_result.sh"
            )
        ),
        None,
    )
    if export is None:
        pytest.skip("cuaworld materials not fetched")
    text = export.read_text()
    assert 'os.system(f"mv {temp_path} {RESULT_FILE}")' in text
    assert "${RESULT_FILE}" not in text


def test_gcompris_live_rollout_artifact_only_false_positives_are_excluded():
    """These GCompris tasks passed in live rollout from terminal-created artifacts
    while the required activity was never used.

    No-op/forged sweeps cannot see this class: the output file must be created
    during a real episode. The exclude is therefore a live-rollout pin, not a
    derived sweep result.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert {
        "science_experiment_catalog",
        "simple_word_processing",
        "solar_system_explore",
        "target_score",
        "vector_drawing_composition",
    } <= {
        task
        for task, reason in excludes["gcompris"].items()
        if reason == "gameable_full"
    }


def test_slicer_live_rollout_missing_ground_truth_verifier_gap_is_excluded():
    """The live trajectory produced measurement/report artifacts, but the verifier
    could not load ground truth and therefore could not score distance or
    classification accuracy. That makes reward=0 unsafe as a default negative."""
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["slicer3d"]["tumor_ventricle_proximity"] == "missing_verifier"


def test_slicer_tracheal_shape_index_loose_pass_is_excluded():
    """B5 live rollout passed despite measuring the wrong slice.

    The task asks for AP/transverse diameters at the aortic arch level. The
    verifier's gate accepts score >= 60 with one accurate diameter, so a report
    from slice 82 passed against GT slice 46. That is a reward-1 false positive.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["slicer3d"]["tracheal_shape_index"] == "gameable_full"


def test_openemr_reschedule_direct_sql_false_positive_is_excluded():
    """B5 live rollout proved the verifier accepts a forged final DB state.

    The agent inserted/deleted appointments directly with SQL, changed duration,
    and did not prove the requested existing appointment record was updated via
    OpenEMR. The pinned verifier still awarded reward=1.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["openemr"]["reschedule_appointment"] == "gameable_full"


def test_openemr_check_in_direct_sql_false_positive_is_excluded():
    """B5 focused reroll passed after a direct SQL status update.

    The trajectory was blocked in the OpenEMR UI, then set ``pc_apptstatus='@'``
    through MySQL. The pinned verifier still awarded reward=1, so this is not a
    trustworthy UI-completion training signal.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["openemr"]["check_in_appointment"] == "gameable_full"


def test_openemr_pinned_setup_and_export_defects_are_excluded():
    """Pinned OpenEMR materials contain setup/export failures outside our schema fix.

    The tasks either target patients/encounters absent from the shipped seed data
    or produce invalid verification JSON after the live setup path. They are
    recorded as live-layer ``other`` entries so offline setup-regeneration does
    not silently drop image-dependent findings.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    for task_id in (
        "add_patient_guarantor",
        "cancel_appointment",
        "close_patient_encounter",
        "document_no_show",
        "document_refusal_of_care",
    ):
        assert excludes["openemr"][task_id] == "other"


def test_b7_live_rollout_reward_signal_defects_are_excluded():
    """B7 visual audit found live false positives/negatives invisible to no-op sweeps.

    These are hand-added pins: they require real agent trajectories, first/middle/last
    screenshots, and final ``05_results.json`` payloads to classify.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["astroimagej"]["identify_variable_star"] == "other"
    assert excludes["astroimagej"]["time_series_subframing"] == "gameable_full"
    assert excludes["openemr"]["add_insurance_info"] == "other"
    assert excludes["openemr"]["document_drug_allergy"] == "other"
    assert excludes["openemr"]["document_fall_risk_assessment"] == "other"
    assert excludes["openemr"]["generate_day_sheet"] == "gameable_full"
    assert (
        excludes["slicer3d"]["split_segment_scissors"]
        == "broken_export_query_live_instance"
    )
    assert excludes["solvespace"]["fishplate_angle_repair"] == "gameable_full"


def test_b8_live_rollout_setup_export_data_and_verifier_defects_are_excluded():
    """B8 visual audit found pinned setup/export/data/verifier defects.

    These are not CUA-Lite code fixes: they are locked upstream task materials where
    the setup did not materialize the promised workspace, the exporter hook name is
    wrong, the loaded medical image cannot support the requested measurement, or a
    malformed-but-agent-created artifact crashes the verifier before it can score 0.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["openemr"]["document_medication_administration"] == "other"
    assert excludes["vscode"]["repair_epidemiological_data_pipeline"] == "setup_aborts"
    assert excludes["slicer3d"]["optic_nerve_sheath_diameter"] == "other"
    assert excludes["slicer3d"]["aorta_measurement"] == "other"
    assert excludes["solvespace"]["slot_profile_tangent"] == "verifier_crash"


def test_b9_live_rollout_reward_data_and_export_defects_are_excluded():
    """B9 visual audit found live false positives and polluted zeroes.

    These pins require real trajectory evidence: no-op/forged sweeps cannot see a
    GCompris target-score false positive, a direct `.slvs` text-edit pass, a test
    hash polluted by pytest bytecode, or a phantom CT paired with generated
    foraminal ground truth.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["gcompris"]["prime_muncher"] == "gameable_full"
    assert excludes["solvespace"]["extrude_constrained_profile"] == "gameable_full"
    assert excludes["vscode"]["repair_historical_nlp_pipeline"] == "other"
    assert excludes["slicer3d"]["neural_foramen_assessment"] == "other"


def test_b10_live_rollout_reward_data_and_verifier_defects_are_excluded():
    """B10 visual audit found more live-only pinned-material defects.

    These are not local adapter fixes: the GCompris clock verifier accepts
    interaction as success, two Slicer3D tasks ship phantom-like volumes for
    clinical prompts, two SolveSpace verifiers text-decode binary `.slvs`
    files before semantic checks, and one VSCode exporter short-circuits hidden
    checks through a brittle mtime gate.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["gcompris"]["clock_activity"] == "gameable_full"
    assert excludes["slicer3d"]["gastric_volume_bariatric"] == "other"
    assert excludes["slicer3d"]["liver_ablation_suitability"] == "other"
    assert excludes["solvespace"]["parametric_scissor_lift_kinematics"] == "other"
    assert excludes["solvespace"]["symmetric_trapezoid_channel"] == "other"
    assert excludes["vscode"]["debug_ml_model_api"] == "other"


def test_b11_live_rollout_reward_signal_defects_are_excluded():
    """B11 visual audit found live-only reward-signal defects.

    These pins require saved trajectories: AstroImageJ reward-1 passes came from
    artifact/numeric outputs rather than trustworthy GUI workflow evidence, and
    GCompris braille visually completed letters that the progress-file verifier
    did not credit.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["astroimagej"]["calibrate_science_frames"] == "gameable_full"
    assert excludes["astroimagej"]["create_color_ratio_map"] == "gameable_full"
    assert excludes["gcompris"]["braille_alphabet"] == "other"


def test_b12_live_rollout_reward_signal_defects_are_excluded():
    """B12 visual audit found live-only false positives and polluted errors.

    These pins require first/middle/last screenshots plus final action/result
    artifacts: no-op or offline forged sweeps cannot see terminal SQL writes,
    GUI-count gaps, or post-task export failures after real agent work.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["astroimagej"]["galaxy_isophotal_morphology"] == "gameable_full"
    assert excludes["astroimagej"]["point_source_suppression"] == "gameable_full"
    assert excludes["astroimagej"]["map_transient_artifacts_zproject"] == "other"
    assert excludes["gcompris"]["reading_readiness_assessment"] == "gameable_full"
    assert excludes["openemr"]["document_flu_vaccine"] == "gameable_full"
    assert excludes["openemr"]["prescribe_medication"] == "gameable_full"
    assert excludes["openemr"]["generate_aging_report"] == "gameable_full"
    assert excludes["openemr"]["add_procedure_fee_sheet"] == "other"


def test_b13_live_rollout_reward_signal_and_task_defects_are_excluded():
    """B13 visual audit found live-only false positives and one impossible task."""
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["astroimagej"]["flat_field_noise_analysis"] == "gameable_full"
    assert excludes["vscode"]["fix_ecommerce_i18n_implementation"] == "gameable_full"
    assert excludes["gcompris"]["railway_construct"] == "other"


def test_b14_live_rollout_reward_signal_and_verifier_defects_are_excluded():
    """B14 visual audit found one AstroImageJ verifier crash and OpenEMR FPs."""
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["astroimagej"]["multiband_histogram_analysis"] == "verifier_crash"
    assert excludes["openemr"]["patient_chart_review"] == "gameable_full"
    assert excludes["openemr"]["document_medical_device"] == "gameable_full"
    assert excludes["openemr"]["generate_audit_log_report"] == "gameable_full"


def test_b15_live_rollout_astroimagej_reward_signal_defects_are_excluded():
    """B15 found AstroImageJ verifier/export defects, not runtime launch failures."""
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["astroimagej"]["eagle_roi_transfer_measurement"] == "gameable_full"
    assert excludes["astroimagej"]["detect_exoplanet_transit"] == "other"
    assert excludes["astroimagej"]["measure_atmospheric_extinction"] == "other"


def test_b16_openemr_direct_state_false_positive_is_excluded():
    """B16 found another OpenEMR reward-1 trajectory solved by direct SQL.

    The agent inserted the requested referral source directly into ``list_options``.
    The pinned verifier checked final DB state and count delta, so the reward is not
    a trustworthy UI-completion training signal.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["openemr"]["add_referral_source_option"] == "gameable_full"


def test_b17d_astroimagej_artifact_only_reward_signal_defects_are_excluded():
    """B17d found AstroImageJ reward-1 passes from terminal artifacts only.

    First/middle/last visual audit showed idle AstroImageJ while the agent
    generated FITS/CSV/ROI/PNG artifacts from a terminal. The verifiers accepted
    the artifacts, and ``plan_mos_spectroscopy_targets`` /
    ``generate_inverted_finding_chart`` feedback explicitly said required visual
    evidence was not detected.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["astroimagej"]["create_binary_source_mask"] == "gameable_full"
    assert (
        excludes["astroimagej"]["generate_inverted_finding_chart"]
        == "gameable_full"
    )
    assert excludes["astroimagej"]["plan_mos_spectroscopy_targets"] == "gameable_full"


def test_b18c_live_rollout_reward_and_material_defects_are_excluded():
    """B18c found live-only reward false positives and polluted zeroes.

    These require saved trajectories: AstroImageJ/Slicer/OpenEMR rewards passed
    despite their own feedback flagging missing or suspicious required fields, and
    two OpenEMR/Slicer zeroes came from pinned material/export defects rather than
    clean agent failures.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["astroimagej"]["empirical_snr_map_generation"] == "gameable_full"
    assert excludes["openemr"]["order_lab_tests"] == "other"
    assert excludes["openemr"]["post_insurance_payment"] == "gameable_full"
    assert excludes["slicer3d"]["clean_segmentation_islands"] == "other"
    assert (
        excludes["slicer3d"]["measure_tumor_vessel_distance"]
        == "gameable_full"
    )


def test_b20_live_rollout_reward_signal_defects_are_excluded():
    """B20 found one GCompris false positive and one VS Code false negative.

    Both require saved trajectories: the GCompris task paid reward 1 before the
    requested five conversions were visibly complete, and the VS Code crypto task
    rejected secure `cryptography` equivalents after tests passed.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["gcompris"]["roman_numerals_study_guide"] == "gameable_full"
    assert excludes["vscode"]["remediate_crypto_flaws"] == "other"


def test_b21_live_rollout_slicer3d_export_false_negative_is_excluded():
    """B21 found a Slicer3D false negative from a task-local export query.

    The final screenshot visibly showed landmark labels and a transform matrix in
    the live Slicer session, but the task's exporter launched a fresh headless
    Slicer and queried an empty scene, producing zero transforms/fiducials.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["slicer3d"]["fiducial_volume_registration"] == "other"


def test_b23_live_rollout_reward_signal_defects_are_excluded():
    """B23 found live-only reward/spec defects after first/middle/last visual audit.

    These pins are not local adapter fixes: AstroImageJ had a robust-sigma prompt
    mismatch and a threshold-too-low false positive, GCompris accepted a forged
    report-only assessment and produced a task-result parse failure, and VSCode
    rejected a functionally correct terminal solution through a hidden VSCode VLM gate.
    """
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )

    assert excludes["astroimagej"]["cosmic_ray_counting"] == "other"
    assert excludes["astroimagej"]["extract_cluster_core"] == "gameable_full"
    assert (
        excludes["gcompris"]["cross_domain_developmental_battery"]
        == "gameable_full"
    )
    assert excludes["gcompris"]["missing_letter_spelling"] == "other"
    assert excludes["vscode"]["repair_cif_parser_library"] == "other"


def test_every_unconditional_pass_verifier_is_excluded():
    """A verifier that can only ever return a pass must carry an exclude_reason.

    Such a stub hands out full reward for any behaviour — a vacuous evaluator, i.e.
    a verifier-side FALSE POSITIVE (not agent-side reward hacking; the agent is not
    involved). 78 of them ship in the pinned registered materials, and every one
    now carries an exclude reason. This test pins that coverage: an absent software
    entry is indistinguishable from a clean one, so a materials bump can silently
    reintroduce a stub. Skips when the gitignored materials are not present (CI).
    """
    import ast

    env_root = _cuaworld_root()
    root = _materials_root()
    tasks = sorted(root.glob("*/*/tasks/*/verifier.py"))
    if not tasks:
        pytest.skip("cuaworld materials not fetched")
    excludes = json.loads((env_root / "data/validation_excludes.json").read_text())

    offenders = []
    for path in tasks:
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        for fn in tree.body:
            if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("verify"):
                continue
            returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
            if len(returns) != 1 or not isinstance(returns[0].value, ast.Dict):
                continue
            try:
                value = ast.literal_eval(returns[0].value)
            except ValueError:
                continue
            if not value.get("passed"):
                continue
            rel = path.relative_to(root).parts
            software, task = rel[0], rel[3]
            # Only REGISTERED tasks matter: 6 stub files ship under tasks/ but are
            # absent from registered.json, so they never enter the registry and can
            # never award anything.
            registered_json = path.parents[1].parent / "registered.json"
            if registered_json.is_file():
                registered = json.loads(registered_json.read_text())
                if not any(task in ids for ids in registered.values() if isinstance(ids, list)):
                    continue
            if not (excludes.get(software) or {}).get(task):
                offenders.append(f"{software}/{task}")

    assert not offenders, (
        "unconditional-pass verifiers with no exclude_reason — they award 1.0 for "
        f"any trajectory: {sorted(offenders)}"
    )
