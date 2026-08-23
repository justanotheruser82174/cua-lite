# Per-software image for a gym-anything (lite.cuaworld.<software>) environment.
#
# Static, committed build recipe: builds on
# cua-lite/lite.cuaworld.base — the shared sandbox Dockerfile built with
# `--build-arg USER=ga`, so the in-container unix user is `ga` (matching the upstream
# gym-anything assets, which hardcode `ga`) — and runs the software's install/setup
# hooks via run_hooks.sh. The build context is assembled by
# scripts/install.sh from the materials repo (cua-lite/lite.cuaworld-assets) at a
# pinned revision, so there is nothing upstream-specific hardcoded here.
#
# Built by:  scripts/install.sh build <software>
#   docker build --build-arg BASE_IMAGE=cua-lite/lite.cuaworld.base:latest \
#     -f docker/Dockerfile -t cua-lite/lite.cuaworld.<software>:latest <context>
# where <context> holds: env.json, scripts/, data/, config/, assets/,
# run_hooks.sh [, post_build.sh]
# Defaults to the ga variant (lite.cuaworld.base) — the fork hooks in run_hooks.sh drop
# to `ga` (su - ga, chown ga:ga), so a bare build must NOT land on the user=user
# sandbox.linux base. install.sh passes this explicitly regardless.
ARG BASE_IMAGE=cua-lite/lite.cuaworld.base:latest
FROM ${BASE_IMAGE}

# SOFTWARE is recorded for provenance; materials freshness is enforced uniformly
# by the lite.src_hash label that install.sh stamps from data/assets.lock.yaml.
ARG SOFTWARE=unknown
LABEL lite.cuaworld.software="${SOFTWARE}"

# gym-anything install hooks assume gnupg (gpg — apt-key for third-party repos) and
# file are present, as the old ga-base baked them. The shared sandbox.linux base
# does not ship them, so restore them here (cuaworld-scoped) — else gpg-using envs
# (e.g. dbeaver) fail at build with "gpg: command not found".
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gnupg file \
 && rm -rf /var/lib/apt/lists/*

# cuaworld has NO agent-parity requirement — it is gym-anything, NOT the OSWorld VM.
# The shared base password-gates `user`'s sudo for OSWorld VM fidelity (osworld/scalecua/
# cuagym); that parity concern does NOT apply to cuaworld. Restore `ga` PASSWORDLESS sudo
# (the gym-anything baseline 2e5168df set `ga ALL=(ALL) NOPASSWD:ALL`) so upstream setup/
# verify hooks — including any that `su - ga` then `sudo` — keep working non-interactively.
RUN echo "ga ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/99-cuaworld-ga \
 && chmod 0440 /etc/sudoers.d/99-cuaworld-ga

COPY . /workspace/
RUN bash /workspace/run_hooks.sh \
 && rm -rf /workspace/run_hooks.sh

# ---------------------------------------------------------------------------
# Runtime gaps the upstream task hooks assume but no image provides. Kept as the
# LAST layer on purpose: every earlier layer (including the multi-minute
# `run_hooks.sh` upstream install) then cache-hits on rebuild, so landing this
# costs ~15-30 s per image instead of a full re-install of all 40.
#
# Tools: measured against what the hooks actually invoke, not a guess. Three
# bases appear below and each figure names its own:
#   ON-DISK    — every task dir under the pinned materials (3216)
#   REGISTERED — the subset a split in registered.json names AND whose task.json
#                parses (3083; what software.py actually builds configs for)
#   LIVE       — REGISTERED minus data/validation_excludes.json (2203; what a
#                rollout samples under the standard exclude_reason filter)
# "present in N images" is the dpkg.log basis: packages this image installed
# STRICTLY BEFORE this final layer, i.e. what the software's own upstream
# installer already brought in. Probed in the 40 built images.
#   jq       appears in 57 `export_result.sh` files over 19 software (ON-DISK);
#            already present in 12 of the 40 images. In the 28 that lack it, 18
#            tasks actually INVOKE it (ON-DISK) — 15 REGISTERED, 10 LIVE, over
#            8 / 5 software. Where it is missing the hook splices an EMPTY
#            command substitution into its JSON heredoc (`"report_content": ,`)
#            so the result file is invalid and the verifier reports "could not
#            retrieve result JSON" on a task the agent completed: 11 of the 18
#            ON-DISK, 9 of the 10 LIVE. The remainder degrade (a `|| echo "[]"`
#            default, or jq used only as a validator that now always says "not
#            JSON" — freecad's 3, which are ON-DISK only, not REGISTERED).
#   bc, sqlite3, rsync, pdftotext (poppler-utils), zenity — same shape, far
#            smaller: 13 more tasks ON-DISK / 12 LIVE (bc 6/5, sqlite3 4/4,
#            rsync 1/1, pdftotext 1/1, zenity 1/1). Most are `|| echo`-guarded
#            and merely degrade; the three UNGUARDED ones are the reason this is
#            worth a layer — eclipse/git_init_and_commit setup_task.sh:27
#            (rsync), vscode/repair_sqlalchemy_orm_layer setup_task.sh:18
#            (sqlite3 seeds the fixture DB), gretl/quantile_regression_food
#            export_result.sh:75,79 (bc computes the ground truth). zenity is
#            the weakest of the five and is kept on one LIVE task only:
#            libreoffice_calc/flooring_estimator setup_task.sh:159 is `|| true`
#            so the hook survives, but without zenity the agent never sees the
#            instruction dialog that setup opens for it.
#            Deliberately NOT added: xmllint / csvtool / yq / 7z (zero hits
#            anywhere in the corpus) and exiftool (its one hit is vlc's own
#            installer apt-getting libimage-exiftool-perl, not a hook call).
#            Also NOT added: vim-common for `xxd`. Its only three call sites are
#            eclipse's `verify_class_file` (a task_utils.sh helper with zero
#            callers) plus blender3d/procedural_rust_material and
#            ugene/sars2_ngs_read_mapping — both in data/validation_excludes.json,
#            so ZERO LIVE tasks. Same for a bare `python` symlink: every apparent
#            `python` invocation in the corpus is inside a heredoc (PyMOL's
#            `python`/`python end` .pml directives, a generated module docstring),
#            never a shell command.
#            Also NOT added: scrot. 32 of the 40 images already get it from their
#            own upstream installer; in the 8 that do not (dbeaver,
#            libreoffice_calc, moodle, odoo, openemr, vlc_media_player, vscode,
#            wordpress) every one of the 384 call sites is `|| true` /
#            `2>/dev/null` guarded, each software's `take_screenshot` helper
#            leads with or falls back to ImageMagick `import -window root` —
#            which the SHARED base ships (lite/gym/sandbox/docker/Dockerfile.linux) —
#            and NO verifier in those 8 reads a hook-written screenshot file at
#            all, so no task scores differently without it.
#            Also NOT added: gdal-bin. 6 gvsig_desktop tasks invoke a GDAL/OGR
#            CLI (5 LIVE) and no other software needs it (qgis's own installer
#            apt-gets gdal-bin, scripts/install_qgis.sh:56). This layer is shared
#            by all 40 images and gdal-bin measures 195 MB installed in the gvsig
#            image, so adding it here costs ~7.8 GB across the set for 5 LIVE
#            tasks in one image. It is also not a failure to record: all 6 of
#            those setup_task.sh `apt-get install -y gdal-bin` themselves, hooks
#            run as ROOT, and the runtime container keeps default bridge
#            networking — verified in cua-lite/lite.cuaworld.gvsig_desktop, the
#            runtime install succeeds and yields GDAL 3.4.1. Only if that runtime
#            apt ever fails do 3 LIVE tasks break: merge_vector_layers
#            (setup_task.sh:42, under `set -e`, so setup aborts),
#            clip_rivers_to_country (export_result.sh:38) and
#            identify_flood_safe_cities (export_result.sh:58); the other two
#            `command -v`-guard their calls.
#
# scipy: astroimagej's upstream installer apt-installs python3-scipy 1.8.0 +
# python3-numpy 1.21.5 and then pip-installs astropy, which drags in numpy 2.2.6
# that SHADOWS the apt numpy — so `import scipy` dies with "compiled using NumPy
# 1.x cannot be run in NumPy 2.2.6". 33 astroimagej tasks import scipy at the TOP
# of a `python3 << PYEOF` heredoc in setup/export (ON-DISK) — 26 REGISTERED, 18
# LIVE. 19 of the 99 astroimagej `setup_task.sh` DO `set -e` (11 of those 33), so
# a third ABORT at the import and the other two thirds run on and exit 0; either
# way the ground truth the heredoc was supposed to write never appears. The
# healthy images (hec_ras, imagej, slicer3d) are already in the state this
# produces: pip scipy alongside numpy 2.x. Guarded on find_spec, so it is a
# no-op on the 32 of 40 images with no scipy at all; only 8 have it
# (astroimagej, freecad, hec_ras, imagej, kstars_sim, pymol, qgis, slicer3d) and
# of those the repair actually pip-installs on 2 — astroimagej and qgis (build
# logs, step #10: `Successfully installed scipy-1.15.3`); the other 6 already
# import cleanly and are left alone.
#
# chmod /workspace/scripts: the upstream materials arrive from the HF snapshot at
# mode 664 and `COPY` preserves it, but task hooks execute `/workspace/scripts/*.sh`
# DIRECTLY — 157 hook files (slicer3d 148: prepare_{brats,amos,lidc,ircadb}_data.sh
# and create_broken_segmentation.sh; dbeaver 5, ardour 2, qgis 1, odoo 1),
# i.e. 135 LIVE tasks. The 5319 `source /workspace/scripts/...` lines are
# correctly NOT counted: sourcing needs no exec bit. Without the chmod they die
# with `Permission denied` and the fixture is never built. It belongs HERE, not
# next to the COPY: an extra RUN before `run_hooks.sh` changes that layer's parent
# and forces a full re-install of all 40 images. Nothing at BUILD time needs the
# bit — run_hooks.sh invokes its hook as `bash <file>`, and every direct exec is
# under tasks/, i.e. runtime. Disjoint from the per-task upload population, which
# lands in /tmp/cuaworld_task_sources and is chmod'd by adapter._run_task_hook.
#
# git safe.directory: post_task hooks inspect repos under /home/ga/workspace after
# the agent has edited/committed in them. In containerized runs those repos can trip
# Git's dubious-ownership guard even though they are the task workspace, producing
# empty commit metadata and false negatives such as vscode/fix_aerospace_telemetry_parser.
# Configure the task-workspace subtree in the image rather than weakening each
# upstream hook or trusting every repository path in the container.
#
# Python modules the hooks import IN THE CONTAINER. Distinct from
# `scripts/install.sh`'s `install_python_deps`, which installs pynrrd/pyshp/nibabel
# on the HOST for the host-side `verifier.py` — a different machine and a different
# interpreter. Hook bodies (`python3 <<'PY' … PY` heredocs and `python3 -c "…"`)
# run HERE, on the session PATH's `python3`, and nothing installed them here.
# Symptom when absent: upstream wraps most of these imports in `try/except`, so the
# hook writes an empty/partial `/tmp/<task>_result.json`, the host verifier scores it
# low, and the run is indistinguishable from the agent failing.
#
# NOTE this is the AGENT's interpreter too — measured in lite.cuaworld.slicer3d,
# the session-PATH `python3`, the `ga` login `python3` and `/usr/bin/python3` are
# one and the same 3.10 (pycharm: one and the same 3.11). `/opt/env/venv` exists but
# no hook uses it. Baking here is therefore agent-visible; accepted deliberately —
# every package below is a read-only artifact parser, and owning a reader does not
# produce a correct segmentation/measurement/reprojection without doing the work.
#
# Per-SOFTWARE on purpose: this layer is shared by all 40 images, so a union install
# would multiply vtk/pandas/gdal across images that never import them (the same
# reasoning that kept gdal-bin out of the shared set above). Bases as elsewhere in
# this file: ON-DISK task dirs / REGISTERED (in a split ∧ task.json parses) / LIVE
# (REGISTERED minus data/validation_excludes.json). Counts are per software, from
# the pinned caches; "GAP" = tasks needing the module that nothing provides at
# runtime. Probed in the 40 built images with each image's own /usr/bin/python3.
#   slicer3d  pynrrd   16 on-disk / 16 reg / 15 LIVE; 10 of the 16 have NO runtime
#                      provider (the other 6 self-heal with `except ImportError:
#                      pip install pynrrd`). 12 of the 18 sites are export_result.sh.
#             pydicom  11 / 11 / 9 task-level, and every one of those self-heals —
#                      but the two SHARED fixture builders do not:
#                      scripts/prepare_ircadb_data.sh:154 imports it UNGUARDED (15
#                      on-disk / 12 LIVE callers, setup aborts → no fixture) and
#                      scripts/prepare_lidc_data.sh:269 imports it GUARDED (29 / 26
#                      LIVE callers, which then silently get placeholder ground
#                      truth). That is the widest blast radius in the corpus.
#             vtk      2 / 2 / 2. decimate_surface_model pip-installs it;
#                      smooth_surface_model does not, and its setup_task.sh import
#                      is unguarded. ~120 MB, one image, so it stays worth it.
#   gvsig     python3-gdal  `from osgeo import ogr` in 5 / 5 / 5 tasks, all
#                      try-guarded, all in BOTH setup_task.sh and export_result.sh.
#                      These 5 do NOT install gdal themselves — the 6 gvsig tasks
#                      that `apt-get install -y -qq gdal-bin` are a DISJOINT set
#                      that only shells out to `ogrinfo`. (Verified in the image:
#                      gdal-bin does pull python3-gdal, so had they overlapped this
#                      would have been a no-op. They do not.) Without it every one
#                      of the 5 export hooks prints `{"error": ..., "tb": ...}` and
#                      the verifier's first subscore (`file_exists`) is absent → 0.
#             pyshp dbfread  8 / 8 / 7 and 1 / 1 / 1, and every site DOES self-
#                      install — but with `pip3 install pyshp … || true`, i.e. a
#                      PyPI round trip on the scoring path whose failure is silent.
#                      0.3 MB total, and the `if ! python3 -c "import shapefile"`
#                      guards then short-circuit, so this is cheaper at runtime too.
#   qgis      pyshp    1 / 1 / 1 (simplify_geometries_web_export) with NO provider.
#   astroimagej  pandas  2/2/2, none self-provided. pip (not apt python3-pandas)
#                      because astroimagej's numpy is the pip 2.2.6 that already
#                      shadows apt's — an apt pandas built against numpy 1.21 would
#                      reproduce the scipy breakage repaired below.
#                cv2     3 on-disk / 2 registered / 1 LIVE:
#                      `generate_inverted_finding_chart` writes
#                      `/tmp/analyze_chart.py` before running it; missing cv2 makes
#                      export_result.sh skip `/tmp/task_result.json` and turns a
#                      visible PNG into reward 0 for an environment reason.
#                      `export_time_series_animation` is excluded and
#                      `generate_finding_chart` is not registered, but the module is
#                      now needed by a live task. pip opencv, not apt, for the same
#                      numpy-ABI reason as pandas/scipy.
#   dbeaver, gretl  pandas  2/2/2 and 1/1/1, none self-provided; the dbeaver and
#                      gretl sites are UNGUARDED.
#   vscode    cv2     2 / 2 / 2. fix_document_image_registration self-installs it
#                      at setup, but debug_software_rasterizer imports cv2 while
#                      generating its setup fixtures with no provider. Bake it here
#                      so setup does not abort for an environment reason.
#   ugene     Pillow   1 / 1 / 1 (plasmid_circular_map_generation), no provider.
#   freecad   FreeCAD  5 / 0 / 0 today: on-disk export hooks invoke FreeCAD's
#                      Python API. Ubuntu's FreeCAD package installs FreeCAD.py
#                      outside python3's default sys.path, while geometry modules
#                      such as Part are safely initialized through freecadcmd.
#                      A per-image .pth keeps upstream hooks unchanged; the Part
#                      probe below uses freecadcmd rather than bare python3
#                      because direct `python3 -c "import Part"` can segfault in
#                      this package mix even though freecadcmd works.
#
# Deliberately NOT installed — verified provided, or zero LIVE tasks:
#   sumo/sumolib          RUNTIME-PROVIDED by SUMO itself. Both hook bodies do
#                         `sys.path.insert(0, $SUMO_HOME/tools)` first, and
#                         /usr/share/sumo/tools/sumolib imports cleanly there
#                         (verified in the image). A pip install would shadow the
#                         app's own copy. Its 1 task is excluded anyway (LIVE 0).
#   vscode/models,utils   AGENT DELIVERABLES. fix_network_provisioning_engine's
#   vscode/cif_parser     export does `sys.path.insert(0, '/home/ga/workspace/
#                         net_provisioner')` and the task is "fix the five logic bugs
#                         in the backend models"; repair_cif_parser_library is the
#                         same shape. ImportError there legitimately means the agent
#                         did not do the work.
#   vscode/ortools        self-installs unconditionally under `set -e` (a failure is
#                         LOUD, not silent) and costs >100 MB for 1 excluded task.
#   pycharm/PIL,cv2       PIL: both tasks excluded (`nonzero_baseline`), LIVE 0.
#                         cv2: debug_document_scanner pip-installs it itself.
#   coppeliasim/…client   4 export tasks, ALL excluded (`missing_verifier`), LIVE 0.
#                         Recorded because it IS broken: the app ships the client at
#                         …/clients/python/src, 3 sites `sys.path.insert` that exact
#                         path, and 2 (rotary_unbalance_vibration_study and
#                         scripts/task_utils.sh `get_scene_object_count`) insert the
#                         parent WITHOUT /src, which holds no such package. The
#                         helper has ZERO callers, so nothing observes it today.
#   slicer3d/skimage      2 tasks, both `pip3 install -q … scikit-image` themselves.
#   solvespace/numpy,trimesh  its 1 task apt+pip-installs both itself.
RUN set -eux; \
    DEBIAN_FRONTEND=noninteractive apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        jq bc sqlite3 rsync poppler-utils zenity; \
    APT_EXTRA=""; PIP_EXTRA=""; IMPORT_CHECK=""; PYTHONPATH_PTH=""; \
    case "${SOFTWARE}" in \
      slicer3d)      PIP_EXTRA="pynrrd pydicom vtk"; IMPORT_CHECK="nrrd pydicom vtk" ;; \
      gvsig_desktop) APT_EXTRA="python3-gdal"; PIP_EXTRA="pyshp dbfread"; \
                     IMPORT_CHECK="osgeo shapefile dbfread" ;; \
      qgis)          PIP_EXTRA="pyshp"; IMPORT_CHECK="shapefile" ;; \
      astroimagej)   PIP_EXTRA="pandas opencv-python-headless"; IMPORT_CHECK="pandas cv2" ;; \
      dbeaver|gretl) PIP_EXTRA="pandas"; IMPORT_CHECK="pandas" ;; \
      vscode)        PIP_EXTRA="opencv-python-headless"; IMPORT_CHECK="cv2" ;; \
      ugene)         PIP_EXTRA="Pillow"; IMPORT_CHECK="PIL" ;; \
      freecad)       PYTHONPATH_PTH="/usr/lib/freecad-python3/lib:/usr/lib/freecad/lib"; \
                     IMPORT_CHECK="FreeCAD" ;; \
    esac; \
    if [ -n "${APT_EXTRA}" ]; then \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ${APT_EXTRA}; \
    fi; \
    rm -rf /var/lib/apt/lists/*; \
    if [ -n "${PIP_EXTRA}" ]; then \
      pip3 install --no-cache-dir ${PIP_EXTRA} \
        || pip3 install --no-cache-dir --break-system-packages ${PIP_EXTRA}; \
    fi; \
    if [ -n "${PYTHONPATH_PTH}" ]; then \
      python3 -c 'import pathlib, site, sys; dirs=[pathlib.Path(p) for p in site.getsitepackages()]; target=next((p for p in dirs if p.exists()), dirs[0]); target.mkdir(parents=True, exist_ok=True); (target / "cuaworld_freecad.pth").write_text(sys.argv[1].replace(":", "\n") + "\n")' "${PYTHONPATH_PTH}"; \
    fi; \
    for m in ${IMPORT_CHECK}; do python3 -c "import ${m}"; done; \
    if [ "${SOFTWARE}" = "freecad" ]; then \
      printf '%s\n' 'import FreeCAD' 'import Part' > /tmp/cuaworld_freecad_probe.py; \
      freecadcmd /tmp/cuaworld_freecad_probe.py >/tmp/cuaworld_freecad_probe.log 2>&1 \
        || { cat /tmp/cuaworld_freecad_probe.log >&2; exit 1; }; \
      rm -f /tmp/cuaworld_freecad_probe.py /tmp/cuaworld_freecad_probe.log; \
    fi; \
    chmod -R a+rx /workspace/scripts; \
    if command -v git >/dev/null; then git config --system --add safe.directory '/home/ga/workspace/*'; fi; \
    if python3 -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("scipy") else 1)'; then \
      python3 -c 'import scipy.ndimage, scipy.optimize, scipy.stats, scipy.signal' >/dev/null 2>&1 \
        || pip3 install --no-cache-dir -U scipy \
        || pip3 install --no-cache-dir --break-system-packages -U scipy; \
    fi
# ``--break-system-packages`` second, not first: these images ship Ubuntu 22.04's
# pip 22.x, which does not know the flag (it arrived with pip 23 / PEP 668) and
# exits 2 on it. Plain install works there; the flag is only needed on an
# externally-managed interpreter, so try the portable form first and fall back.
# The trailing `echo` keeps the whole chain non-fatal under ``set -e``: a repair
# that cannot reach an index is a degraded astroimagej/qgis, not a reason an
# image that used to build stops building.
#
# The `IMPORT_CHECK` loop above is the OPPOSITE and deliberately so: it has no
# fallback, so a `PIP_EXTRA` that silently did not land FAILS THE BUILD. These
# packages exist to stop a missing module from becoming a silent partial score, so
# shipping an image that quietly lacks one would recreate the exact defect. A build
# that stops is loud; a rollout that scores 0 for an environment reason is not.
