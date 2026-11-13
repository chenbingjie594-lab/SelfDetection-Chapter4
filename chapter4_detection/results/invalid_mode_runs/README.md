# Invalid ablation runs (excluded)

The two seed-42 runs in this directory are retained only for auditability and
must not be used in tables or conclusions. Their training checkpoints reload
with `NBRDE.mode == "full"`, although the requested experiments were
`no_reference` and `no_gate`. Ultralytics reconstructed the network from the
Full model YAML and discarded the runtime-only mode assignment.

The correction uses a dedicated model YAML for each ablation so the mode is
part of the serialized model topology. Corrected runs use the original public
run names after this quarantine.
