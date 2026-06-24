"""Feature-agnostic tuning SCHEMA + preset I/O.

A feature's `tuning.py` describes its fittable constants once — a `FIELDS` map of
`name -> _F(doc, kind)` (the canonical commented schema; JSON can't carry comments)
and a `GROUPS` tree (the nested layout a preset JSON is written in) — and hands them
to `TuningSchema`. The schema then provides:

  - `.Tuning`        an immutable namedtuple (one field per constant; docstring built
                     from FIELDS) — the dependency-injected config the analysis reads.
  - `.load(preset)`  read + validate a preset (NAME or path; nested or flat JSON).
  - `.dump(cfg)`     write a Tuning back as a nested preset (the calibration runner
                     emits a drop-in `fitted_preset.json` this way).

Why split values out: a preset is a plain `name -> value` JSON map, so adopting a
calibration result is "drop in a new preset file", never editing Python. The per-field
RATIONALE (the valuable part, distilled from tuning sessions) stays HERE in FIELDS.

Structural/correctness constants (byte offsets, blob templates, mask versions) do NOT
belong in a schema — a "different" value there is a corrupt sidecar, not a different
look — so each feature keeps those inline in its source.
"""
import collections
import json
import os

_Field = collections.namedtuple("_Field", "kind doc")


def _F(doc, kind="float"):
    """A schema entry. `kind` drives load-time coercion so a JSON `2` stays an int
    and a JSON `[5, 30]` becomes the `(5.0, 30.0)` tuple the code expects."""
    return _Field(kind, doc)


class TuningSchema:
    """Holds a feature's FIELDS + GROUPS and serves its `Tuning` type + preset I/O.

    `fields`  OrderedDict name -> _F(doc, kind in {float,int,tuple}).
    `groups`  OrderedDict kind -> OrderedDict sub_stage -> [field names]; purely a
              cosmetic layout for the preset JSON (load flattens it). MUST partition
              FIELDS exactly (asserted) so a new constant can't be silently dropped.
    `presets_dir`  directory bare preset NAMEs resolve into (NAME -> NAME.json).
    """

    def __init__(self, fields, groups, presets_dir, type_name="Tuning"):
        self.FIELDS = fields
        self.GROUPS = groups
        self.PRESETS_DIR = presets_dir
        self.INT_FIELDS = frozenset(n for n, f in fields.items() if f.kind == "int")
        self.TUPLE_FIELDS = frozenset(n for n, f in fields.items()
                                      if f.kind == "tuple")
        self.BOOL_FIELDS = frozenset(n for n, f in fields.items()
                                     if f.kind == "bool")

        # The grouping must cover EXACTLY the field set, once each.
        grouped = [n for kind in groups.values()
                   for grp in kind.values() for n in grp]
        assert sorted(grouped) == sorted(fields) and len(grouped) == len(fields), (
            "GROUPS must partition FIELDS exactly: "
            f"missing {sorted(set(fields) - set(grouped))}, "
            f"extra/dup {sorted([n for n in grouped if grouped.count(n) > 1 or n not in fields])}")

        self.Tuning = collections.namedtuple(type_name, list(fields))
        self.Tuning.__doc__ = (
            "Immutable tuning configuration (one field per fittable constant; values "
            "loaded from a preset JSON). Field docs:\n\n"
            + "\n".join(f"  {n}: {f.doc}" for n, f in fields.items()))

    # -- coercion ----------------------------------------------------------
    def _coerce(self, name, val):
        if name in self.BOOL_FIELDS:
            return bool(val)
        if name in self.INT_FIELDS:
            return int(round(val)) if isinstance(val, float) else int(val)
        if name in self.TUPLE_FIELDS:
            return tuple(float(x) for x in val)
        return float(val)

    def from_mapping(self, raw, source="<mapping>"):
        """Build a Tuning from a flat name->value mapping, validating it covers
        EXACTLY the schema (no missing or unknown keys) and coercing types."""
        missing = [n for n in self.FIELDS if n not in raw]
        unknown = [n for n in raw if n not in self.FIELDS]
        if missing or unknown:
            raise ValueError(
                f"preset {source}: missing {missing or '[]'}, unknown {unknown or '[]'}")
        return self.Tuning(**{n: self._coerce(n, raw[n]) for n in self.FIELDS})

    def _flatten(self, tree, source, _flat=None):
        """Collapse a (possibly nested) preset mapping to a flat name->value dict.
        Any key that is a field name is a leaf; any other key must map to a nested
        dict (a group) and is recursed into. Accepts the OLD flat layout unchanged.
        Raises on a stray/unknown key."""
        flat = {} if _flat is None else _flat
        for k, v in tree.items():
            if k in self.FIELDS:
                if k in flat:
                    raise ValueError(f"preset {source}: field {k!r} appears twice")
                flat[k] = v
            elif isinstance(v, dict):
                self._flatten(v, source, flat)
            else:
                raise ValueError(
                    f"preset {source}: {k!r} is neither a tuning field nor a group")
        return flat

    # -- paths + load/dump -------------------------------------------------
    def preset_path(self, preset):
        """Resolve a preset selector to a JSON path. A bare NAME -> presets/NAME.json;
        a value containing a path separator or ending in .json is used verbatim."""
        if (os.sep in preset or (os.altsep and os.altsep in preset)
                or preset.endswith(".json")):
            return preset
        return os.path.join(self.PRESETS_DIR, preset + ".json")

    def load(self, preset="default"):
        """Load + validate a preset (NAME or path) into an immutable Tuning. The
        JSON may be nested by GROUPS or a flat name->value map (both accepted)."""
        path = self.preset_path(preset)
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return self.from_mapping(self._flatten(raw, path), source=path)

    def to_mapping(self, cfg):
        """Tuning -> a FLAT JSON-ready dict (tuples become lists)."""
        out = collections.OrderedDict()
        for n in self.FIELDS:
            v = getattr(cfg, n)
            out[n] = list(v) if isinstance(v, tuple) else v
        return out

    def to_nested(self, cfg):
        """Tuning -> a NESTED JSON-ready dict grouped by GROUPS (tuples -> lists)."""
        flat = self.to_mapping(cfg)
        out = collections.OrderedDict()
        for kind, subs in self.GROUPS.items():
            out[kind] = collections.OrderedDict(
                (sub, collections.OrderedDict((n, flat[n]) for n in names))
                for sub, names in subs.items())
        return out

    def dump(self, cfg, path):
        """Write a Tuning as a NESTED preset JSON (values only; docs live in the
        feature's tuning.py). Used by the calibration runner to emit a fitted preset."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_nested(cfg), fh, indent=2)
            fh.write("\n")
