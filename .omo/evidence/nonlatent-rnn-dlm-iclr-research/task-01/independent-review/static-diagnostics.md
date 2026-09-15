# Static diagnostics

Run through the available LSP without installing tools.

- `lsp_status`: 42 servers configured; only `pyright` installed. `basedpyright` and `ruff` are missing.
- `scale/experiments/nonlatent_iclr/service.py`: no diagnostics.
- `scale/experiments/nonlatent_iclr/cli.py`: no diagnostics.
- `scale/tests/nonlatent_iclr/test_audit_cli.py`: `error[Pyright] (reportMissingImports) at 12:5: Import "experiments.nonlatent_iclr.service" could not be resolved`.

The runtime tests pass because the test mutates `sys.path`; that does not make the LSP diagnostic clean. This review records the project-layout/configuration gap rather than waiving it.
