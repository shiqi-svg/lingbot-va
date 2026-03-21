# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist


def _get_supported_cuda_arches():
    arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
    supported_arches = []
    for arch in arch_list:
        if not arch.startswith("sm_"):
            continue
        suffix = arch[3:]
        if suffix.isdigit():
            supported_arches.append(int(suffix))
    return sorted(set(supported_arches))


def ensure_cuda_compatibility(device_index):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please verify the PyTorch CUDA installation.")

    device = torch.device(f"cuda:{device_index}")
    try:
        sample = torch.randn((4, 4), device=device)
        _ = (sample @ sample).sum().item()
        return
    except Exception as smoke_test_error:
        smoke_test_message = str(smoke_test_error)

    supported_arches = _get_supported_cuda_arches()
    if not supported_arches:
        raise RuntimeError(
            f"CUDA initialization failed on device cuda:{device_index}: {smoke_test_message}"
        ) from smoke_test_error

    major, minor = torch.cuda.get_device_capability(device_index)
    current_arch = int(f"{major}{minor}")
    device_name = torch.cuda.get_device_name(device_index)
    supported_text = ", ".join(f"sm_{arch}" for arch in supported_arches)
    torch_cuda = torch.version.cuda or "unknown"
    raise RuntimeError(
        f"Detected GPU '{device_name}' with compute capability sm_{current_arch}, "
        f"but the current PyTorch build ({torch.__version__}, CUDA {torch_cuda}) only supports "
        f"{supported_text}. A CUDA smoke test on this device also failed with: {smoke_test_message}. "
        "Install a PyTorch build that can execute CUDA kernels on this GPU."
    ) from smoke_test_error


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True):
    """
    TODO
    """
    if eval_mode:
        model.eval().requires_grad_(False)
    if dist.is_initialized():
        dist.barrier()

    if dist.is_initialized():
        model = shard_fn(model)
    else:
        model.to(param_dtype)
        model.to(device)

    return model


def init_distributed(world_size, local_rank, rank):
    ensure_cuda_compatibility(local_rank)
    torch.cuda.set_device(local_rank)
    if world_size <= 1:
        return
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            rank=rank,
                            world_size=world_size)

def dist_mean(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.AVG)
    return local_tensor

def dist_max(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)
    return local_tensor
