# Invalid-run record

The following local-only runs are excluded from every statistic and ignored by Git:

- `amp_dtype_prebatch`: detached diagnostic recomputation initially used a float/half mismatch and stopped before the first batch.
- `missing_assignment_call`: trace lacked a unique assignment-call key, so different batches could not be paired safely.
- `detached_process_reaped`: background execution was interrupted by the host/session lifecycle.
- `missing_assignment_call_old_process`: an older host process continued writing the pre-fix trace and was stopped explicitly.

Only `runs/e0_fold0_seed42_geometry_diagnostic` with protocol `stage14_geometry_diagnostic_v2`, `completed=true`, 6,625 trace rows, and the `assignment_call` column is formal evidence.
