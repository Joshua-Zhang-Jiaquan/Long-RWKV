from __future__ import annotations

from datetime import timedelta
import os
import subprocess
from typing import Final

from .contracts import GPURecord, QualificationRuntimeError
from .controls import ProbeConfig


_AUTHORIZED_H100_NAMES: Final = frozenset(
    {"NVIDIA H100 80GB HBM3", "NVIDIA H100-SXM5-80GB"}
)
_MINIMUM_H100_80GB_MEMORY_MIB: Final = 79_000
_MAXIMUM_H100_80GB_MEMORY_MIB: Final = 83_000


def is_authorized_h100(name: str, memory_mib: int) -> bool:
    return (
        name.strip() in _AUTHORIZED_H100_NAMES
        and _MINIMUM_H100_80GB_MEMORY_MIB
        <= memory_mib
        <= _MAXIMUM_H100_80GB_MEMORY_MIB
    )


def initialize_hardware(config, rank, torch_module, dist_module):
    if not torch_module.cuda.is_available():
        raise QualificationRuntimeError("cuda_unavailable")
    if torch_module.cuda.device_count() != config.expected_world_size:
        raise QualificationRuntimeError("visible_device_count_not_8")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if rank not in range(8) or local_rank not in range(8):
        raise QualificationRuntimeError("distributed_rank_identity_invalid")
    torch_module.cuda.set_device(local_rank)
    device = torch_module.device("cuda", local_rank)
    torch_module.cuda.set_per_process_memory_fraction(config.memory_fraction, device)
    process_timeout = config.nccl_timeout_seconds
    dist_module.init_process_group(
        backend="nccl", timeout=timedelta(seconds=process_timeout)
    )
    if (
        dist_module.get_world_size() != config.expected_world_size
        or dist_module.get_rank() != rank
    ):
        raise QualificationRuntimeError("distributed_topology_mismatch")
    inventory = gpu_inventory()
    if len(inventory) != config.expected_world_size:
        raise QualificationRuntimeError("gpu_inventory_count_not_8")
    if any(not is_authorized_h100(gpu.name, gpu.memory_mib) for gpu in inventory):
        raise QualificationRuntimeError("authorized_h100_80g_shape_not_observed")
    if len({gpu.uuid for gpu in inventory}) != config.expected_world_size:
        raise QualificationRuntimeError("gpu_uuid_identity_not_unique")
    physical_bytes = torch_module.cuda.get_device_properties(device).total_memory
    return local_rank, device, inventory, physical_bytes


def gpu_inventory() -> tuple[GPURecord, ...]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    records: list[GPURecord] = []
    for line in result.stdout.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 3:
            raise QualificationRuntimeError("nvidia_smi_inventory_row_malformed")
        records.append(
            GPURecord(name=fields[0], uuid=fields[1], memory_mib=int(fields[2]))
        )
    return tuple(records)
