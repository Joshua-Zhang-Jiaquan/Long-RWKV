# Adapter-v6 post-QA validation correction

## Recorded deviation

After the fully compliant CPU QA rerun, the parent-side evidence check ran:

```bash
python -m json.tool "qa-rerun.json" >/dev/null && python -m json.tool "verdict.json" >/dev/null
```

This helper omitted `PYTHONDONTWRITEBYTECODE`, an external `PYTHONPYCACHEPREFIX`, and `PYTHONSAFEPATH`; it is not represented as compliant. The recorded command names only the two superseding-01 JSON inputs, resolves the standard-library `json.tool` module, and redirects rendered JSON to `/dev/null`. It names no staged-release, source, test, checkpoint, model, or generated-output path. No syscall trace exists, so no stronger access claim is made.

## Corrected validation

A fresh scratch cache was created at `/tmp/adapter-v6-json-correction.X0Dk1p`. Only the affected evidence validation was repeated:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/adapter-v6-json-correction.X0Dk1p PYTHONSAFEPATH=1 /usr/bin/python -P -m json.tool "qa-rerun.json" >/dev/null
env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/adapter-v6-json-correction.X0Dk1p PYTHONSAFEPATH=1 /usr/bin/python -P -m json.tool "verdict.json" >/dev/null
```

Both commands exited 0. The prior 91/91 CPU QA scope was not repeated.

## Preservation and cleanup

Shell-only checks returned:

```text
tree=5b86080fba3c7ed11c65c9b7045349f0ad074973921ff04c22347ac4db8690f9
manifest=692d1cc88d6b2c6335736d47492aa933bf57b9a23a4d0dcc9e5e3eeda165cf32
writable_paths=0
symlinks=0
bytecode_paths=0
scratch_cache_entries=0
```

The review-created scratch cache was then removed. No frozen v5/v6 payload or consumed qualification05 identity was changed.
