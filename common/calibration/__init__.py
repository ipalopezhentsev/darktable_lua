"""Shared, feature-agnostic calibration machinery.

The two darktable plugins (auto_negadoctor, auto_retouch) both calibrate algorithm
constants against hand-annotated ground truth. The MECHANICS — externalising the
constants to `presets/*.json`, enumerating the fittable subset per "kind", the
optimizers, and the recorded-session bookkeeping — are identical between
them. They live here; each feature supplies only its schema (the constants + docs),
its registry (the fittable menu) and its per-kind EVALUATORS (the metric).

This mirrors `common/debug_ui_base.py`: a shared base + thin feature subclasses.

  - schema.py    TuningSchema  — preset load/dump + the immutable `Tuning` namedtuple
  - registry.py  Registry      — the fittable catalog + per-trial cfg builder
  - runner.py    run_main()    — optimizers + recorded sessions, driven by an Adapter
"""
