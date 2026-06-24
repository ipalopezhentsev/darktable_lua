"""Feature-agnostic calibration REGISTRY: the catalog of fittable constants.

A feature's calibration registry module describes, per calibration KIND, which
constants that kind may fit and over what range/grid step (the "menu"; a session's
config picks a SUBSET). It builds a nested `REGISTRY` dict with the small `P()`
helper, then hands it to `Registry` here, which provides the generic operations the
runner needs:

  - `fittable(kind)` / `kind_of(name)`     enumerate the menu
  - `current(name)` / `snapshot` / `apply` / `restore`   live-module getattr/setattr
    (the legacy global path — kept as utilities; the runner uses to_tuning instead)
  - `to_tuning(overrides)`   build an IMMUTABLE per-trial `Tuning` (base + overrides),
    so independent trials never mutate shared module state and can run in parallel.

Tuple-valued constants are addressed PER ELEMENT as `NAME[i]`; integer constants
carry `"int": True` in their spec. `to_tuning` parses both.
"""


def P(lo, hi, step, integer=False, module=None):
    """A fittable-constant spec: search `range`, `grid_step`, the owning `module`
    (where the live value lives — usually the feature's primary analysis module),
    and an `int` flag. Features wrap this with their own default `module`."""
    e = {"module": module, "range": [lo, hi], "grid_step": step}
    if integer:
        e["int"] = True
    return e


class Registry:
    """Wraps a feature's REGISTRY dict + the modules its constants live on.

    `registry`        {kind: {name: spec}} (spec from `P`).
    `modules`         {module_name: module_object} the constants live on.
    `schema`          the feature's TuningSchema (its `.Tuning` type).
    `primary_module`  module name whose `DEFAULT_TUNING` is the to_tuning base and
                      whose attributes a Tuning field maps to (asserted in to_tuning).
    """

    def __init__(self, registry, modules, schema, primary_module):
        self.REGISTRY = registry
        self._MODULES = modules
        self._schema = schema
        self._primary = primary_module

    def fittable(self, kind):
        """The {name: spec} the given kind may fit (copy; safe to mutate)."""
        return {n: dict(s) for n, s in self.REGISTRY.get(kind, {}).items()}

    def _find(self, name):
        for kind in self.REGISTRY.values():
            if name in kind:
                return kind[name]
        raise KeyError(f"{name!r} is not a registered fittable constant")

    def kind_of(self, name):
        for kind, params in self.REGISTRY.items():
            if name in params:
                return kind
        raise KeyError(name)

    @staticmethod
    def _parse(name):
        """('WB_HIGH_PRIOR[0]') -> ('WB_HIGH_PRIOR', 0); ('P_LOW') -> ('P_LOW', None)."""
        if name.endswith("]") and "[" in name:
            base, idx = name[:-1].split("[", 1)
            return base, int(idx)
        return name, None

    @staticmethod
    def _coerce(val, spec):
        return int(round(val)) if spec.get("int") else float(val)

    def current(self, name):
        """The constant's LIVE value on its module (the search start / init). For an
        indexed name, the addressed tuple element."""
        spec = self._find(name)
        base, idx = self._parse(name)
        val = getattr(self._MODULES[spec["module"]], base)
        return float(val[idx]) if idx is not None else float(val)

    def snapshot(self, names):
        """Capture the live value of each name's BASE attribute (whole tuple for
        indexed names) so it can be restored intact. Keyed by (module, base)."""
        snap = {}
        for name in names:
            spec = self._find(name)
            base, _ = self._parse(name)
            key = (spec["module"], base)
            if key not in snap:
                snap[key] = getattr(self._MODULES[spec["module"]], base)
        return snap

    def apply(self, overrides):
        """Set each name on its module. Indexed names of the same tuple are gathered
        and the tuple is rebuilt once (tuples are immutable)."""
        tuples = {}   # (module, base) -> {idx: value}
        for name, val in overrides.items():
            spec = self._find(name)
            base, idx = self._parse(name)
            if idx is None:
                setattr(self._MODULES[spec["module"]], base, self._coerce(val, spec))
            else:
                tuples.setdefault((spec["module"], base), {})[idx] = self._coerce(val, spec)
        for (modname, base), idxvals in tuples.items():
            cur = list(getattr(self._MODULES[modname], base))
            for i, v in idxvals.items():
                cur[i] = v
            setattr(self._MODULES[modname], base, tuple(cur))

    def restore(self, snap):
        """Inverse of apply(): write the snapshotted base values back."""
        for (modname, base), val in snap.items():
            setattr(self._MODULES[modname], base, val)

    def to_tuning(self, overrides, base=None):
        """Build an immutable `Tuning` = `base` (default the primary module's
        DEFAULT_TUNING) with `overrides` applied — the THREAD-SAFE alternative to
        apply()/restore(): the runner gives each trial its OWN cfg to pass into the
        analysis functions instead of mutating shared module globals, so independent
        trials can run in parallel. Indexed names (NAME[i]) patch the addressed tuple
        element; ints are rounded. Every fittable constant must live in the primary
        module (asserted), so it is a Tuning field."""
        if base is None:
            base = getattr(self._MODULES[self._primary], "DEFAULT_TUNING")
        fields = base._asdict()
        tuples = {}   # base name -> {idx: value}
        for name, val in overrides.items():
            spec = self._find(name)
            if spec["module"] != self._primary:
                raise ValueError(
                    f"{name!r} is not in {self._primary}; cfg only covers "
                    f"{self._primary} constants")
            b, idx = self._parse(name)
            if idx is None:
                fields[b] = self._coerce(val, spec)
            else:
                tuples.setdefault(b, {})[idx] = self._coerce(val, spec)
        for b, idxvals in tuples.items():
            cur = list(fields[b])
            for i, v in idxvals.items():
                cur[i] = v
            fields[b] = tuple(cur)
        return self._schema.Tuning(**fields)
