# Unambiguous configuration mappings

Experiment, sweep, refit, base-protocol and SPH transport loaders reject repeated
explicit mapping keys. Refit JSON applies the same rule at every object depth.
A duplicate is refused even when both values are equal: there must be one
visible declaration of each setting. The existing configuration-specific error
family is preserved, with the parser's duplicate-key error as its cause.

For example, this development setting is invalid:

```yaml
optimization:
  epochs: 20
  epochs: 30
```

Previously the YAML parser silently kept the later value before the typed
configuration validator saw it. The shared decoder now rejects the file before
it can become a resolved configuration or supply a run's settings.
[The YAML mapping specification requires unique keys](https://yaml.org/type/map.html).
This is not a change to PyYAML's scalar-resolution rules or a YAML-version migration.

Valid anchors, aliases and merge-based defaults keep their previous meaning.
An explicit key may override a merged default. The check runs on each original
mapping before merge expansion, including an inline mapping used only as a
merge source. Repeated explicit `<<` clauses are rejected; use one merge clause
with a sequence of sources. Keys that collide in Python's resulting mapping
are also refused. Other SafeLoader restrictions, including refusal of Python
object constructors and unhashable keys, remain in force.

The implementation is local to
[`_config_loading.py`](../src/ecg_trust/_config_loading.py). It does not change
PyYAML's global SafeLoader or other packages' parsing behavior. Source-calibration
and OOD pipelines retain their existing duplicate-key/frozen-byte checks.

All committed YAML files decode to the same values, and valid protocol,
experiment, sweep and refit hashes remain unchanged. Frozen files are not
rewritten or normalized. Correct a malformed working configuration explicitly;
do not edit a sealed protocol or reinterpret a prior run's identity. The
[reproduction workflow](REPRODUCIBILITY.md) still governs scientific execution.
