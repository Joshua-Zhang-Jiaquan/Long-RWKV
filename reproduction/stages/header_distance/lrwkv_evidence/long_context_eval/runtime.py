"""Hardware/software provenance for measured request costs."""
from importlib.metadata import PackageNotFoundError, version


def metadata(torch, device):
    properties = torch.cuda.get_device_properties(device)
    packages = {}
    for name in ('torch', 'triton', 'transformers', 'flash-linear-attention', 'fla-core'):
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = None
    # NVIDIA images may vendor Triton without a distribution named "triton".
    if packages['triton'] is None:
        import triton
        packages['triton'] = getattr(triton, '__version__', None)
    return dict(gpu_name=properties.name, gpu_total_memory_bytes=properties.total_memory,
                gpu_multiprocessors=properties.multi_processor_count,
                compute_capability=list(torch.cuda.get_device_capability(device)),
                cuda_runtime=torch.version.cuda, packages=packages)
