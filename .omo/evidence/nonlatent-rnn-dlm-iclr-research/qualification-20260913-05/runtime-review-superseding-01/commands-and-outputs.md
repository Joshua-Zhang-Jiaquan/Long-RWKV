# Focused commands and exact outputs

All commands were local and read-only. No model was constructed, CUDA was not initialized, and `CUDA_VISIBLE_DEVICES` was empty for the Torch import.

## PEP 440 normalization

Command used `/usr/bin/python`, `packaging.version.Version`, and installed packaging `23.2` to parse both raw strings.

```json
{"equal": true, "input_metadata": "2.8.0a0+5228986c39.nv25.6", "input_module": "2.8.0a0+5228986c39.nv25.06", "metadata_local_segments": ["5228986c39", "nv25", "6"], "module_local_segments": ["5228986c39", "nv25", "6"], "normalized_metadata": "2.8.0a0+5228986c39.nv25.6", "normalized_module": "2.8.0a0+5228986c39.nv25.6", "packaging_distribution_version": "23.2"}
```

## CPU-only installed Torch observation

```json
{"cuda_visible_devices": "", "distribution_metadata": [{"path": "/usr/local/lib/python3.12/dist-packages/torch-2.8.0a0+5228986c39.nv25.6.dist-info/METADATA", "sha256": "f275a4884d5042cfcd97cdf69ba164a102664e0cfea8dee26db4a63243ff3c68", "size_bytes": 26954}], "python_executable": "/usr/bin/python", "python_version": "3.12.3", "torch_cuda_initialized": false, "torch_distribution_name": "torch", "torch_distribution_version": "2.8.0a0+5228986c39.nv25.6", "torch_file": "/usr/local/lib/python3.12/dist-packages/torch/__init__.py", "torch_module_version": "2.8.0a0+5228986c39.nv25.06", "torch_serialization_py": {"path": "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", "sha256": "8263ad671446bb287b1d343309dfa7dcd4aa12ec37e89498b2522bf840e905e0", "size_bytes": 84598}, "torch_version_cuda": "12.9", "torch_version_py": {"path": "/usr/local/lib/python3.12/dist-packages/torch/version.py", "sha256": "f705f2955ebf304eab692c52e648caa82cfb45ae849979f74bc7ffde19706bd2", "size_bytes": 263}}
```

## Worker runtime-environment extraction

```json
{"python_executable": "/usr/bin/python", "python_version": "3.12.3", "runtime_modules_loaded": [], "status": "MATCH", "torch_package": {"distribution": "torch", "expected_version": "2.8.0a0+5228986c39.nv25.6", "observed_version": "2.8.0a0+5228986c39.nv25.6", "status": "MATCH"}, "torch_serialization_source": {"expected_sha256": "8263ad671446bb287b1d343309dfa7dcd4aa12ec37e89498b2522bf840e905e0", "expected_size_bytes": 84598, "observed_sha256": "8263ad671446bb287b1d343309dfa7dcd4aa12ec37e89498b2522bf840e905e0", "observed_size_bytes": 84598, "requested_path": "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", "resolved_path": "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", "status": "MATCH"}}
```

## Source hashes

```text
1c9bd38b71dc5b22d305fe65ac5c6403f08443f449bd0f58e6a550835823ce65  manifest.py
aeb15fe3ef7c6f8b81570a8e9c86cc773eb77b23091bd115fb4c89489acf3a69  runtime_environment.py
c34753e6ed0450f8697a9fda761cc3a3571b3c3c51e4d1d159459a4b98cb6dde  runtime_checks.py
112ab2595a9db913a40c3976a96998ccf54f9fb7a9680da1ed841cf871dbd52b  run_qualification.sh
```
