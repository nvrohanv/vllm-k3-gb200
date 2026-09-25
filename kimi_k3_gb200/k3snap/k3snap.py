"""vLLM plugin: `--load-format k3snap` snapshots post-processed weights per rank.

First start (no snapshot): load the checkpoint with fastsafetensors as usual,
then save every tensor the model holds (parameters, buffers and plain tensor
attributes, after process_weights_after_loading) to $K3SNAP_DIR/<key>/rank<r>.
Later starts: build the model, run process_weights_after_loading on
uninitialized memory to recreate the final layout, then copy the snapshot into
those tensors in place. Any missing tensor or shape/dtype mismatch is fatal.
"""

import concurrent.futures
import hashlib
import json
import os
import time

import torch
from torch import nn

SNAP_ROOT = os.environ.get("K3SNAP_DIR", "/local/k3snap")
PART_BYTES = 4 << 30


def _iter_tensors(model: nn.Module):
    """Yield (key, tensor) for every tensor reachable from each module."""
    for mod_name, mod in model.named_modules():
        seen = set()
        for attr, value in list(mod._parameters.items()) + list(mod._buffers.items()):
            if value is not None:
                seen.add(attr)
                yield f"{mod_name}::{attr}", value.data if isinstance(value, nn.Parameter) else value
        for attr, value in vars(mod).items():
            if attr in seen or attr.startswith("_"):
                continue
            if isinstance(value, torch.Tensor):
                yield f"{mod_name}::{attr}", value
            elif isinstance(value, (list, tuple)) and value and all(isinstance(v, torch.Tensor) for v in value):
                for i, v in enumerate(value):
                    yield f"{mod_name}::{attr}[{i}]", v


def _snapshot_dir(vllm_config) -> str:
    import vllm
    from vllm.distributed import get_tensor_model_parallel_rank

    pc = vllm_config.parallel_config
    key_src = json.dumps({
        "model": vllm_config.model_config.model,
        "vllm": vllm.__version__,
        "tp": pc.tensor_parallel_size,
        "pp": pc.pipeline_parallel_size,
        "ep": pc.enable_expert_parallel,
        "moe_backend": str(vllm_config.kernel_config.moe_backend),
        "extra": os.environ.get("K3SNAP_KEY", ""),
        # Flags that change which tensors the model holds after processing.
        "kda6": os.environ.get("K3OPT_KDA6", "0"),
    }, sort_keys=True)
    key = hashlib.sha1(key_src.encode()).hexdigest()[:12]
    return os.path.join(SNAP_ROOT, key, f"rank{get_tensor_model_parallel_rank()}"), key_src


def _save(model: nn.Module, path: str) -> None:
    from safetensors.torch import save_file

    t0 = time.perf_counter()
    os.makedirs(path, exist_ok=True)
    manifest, part, part_bytes, part_idx = {}, {}, 0, 0

    def flush():
        nonlocal part, part_bytes, part_idx
        if part:
            save_file(part, os.path.join(path, f"part-{part_idx:04d}.safetensors"))
            part, part_bytes, part_idx = {}, 0, part_idx + 1

    for key, t in _iter_tensors(model):
        manifest[key] = {"shape": list(t.shape), "dtype": str(t.dtype), "part": part_idx}
        part[key] = t.detach().contiguous().cpu()
        part_bytes += t.numel() * t.element_size()
        if part_bytes >= PART_BYTES:
            flush()
    flush()
    with open(os.path.join(path, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    open(os.path.join(path, "COMPLETE"), "w").close()
    print(f"[k3snap] saved {len(manifest)} tensors to {path} in {time.perf_counter() - t0:.1f}s", flush=True)


def _load(model: nn.Module, path: str) -> None:
    from safetensors import safe_open

    t0 = time.perf_counter()
    with open(os.path.join(path, "manifest.json")) as f:
        manifest = json.load(f)
    targets = dict(_iter_tensors(model))
    missing = sorted(set(targets) - set(manifest))
    extra = sorted(set(manifest) - set(targets))
    if missing or extra:
        raise RuntimeError(f"[k3snap] tensor set mismatch: missing={missing[:5]} extra={extra[:5]}")
    for key, t in targets.items():
        meta = manifest[key]
        if list(t.shape) != meta["shape"] or str(t.dtype) != meta["dtype"]:
            raise RuntimeError(f"[k3snap] {key}: {tuple(t.shape)}/{t.dtype} vs snapshot {meta}")
    by_part = {}
    for key, meta in manifest.items():
        by_part.setdefault(meta["part"], []).append(key)
    device = torch.cuda.current_device()

    def load_part(idx):
        torch.cuda.set_device(device)
        nbytes = 0
        with safe_open(os.path.join(path, f"part-{idx:04d}.safetensors"), framework="pt", device="cpu") as f:
            for key in by_part[idx]:
                src = f.get_tensor(key)
                targets[key].copy_(src)
                nbytes += src.numel() * src.element_size()
        return nbytes

    with concurrent.futures.ThreadPoolExecutor(int(os.environ.get("K3SNAP_THREADS", "8"))) as pool:
        total = sum(pool.map(load_part, sorted(by_part)))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[k3snap] loaded {len(manifest)} tensors ({total / 2**30:.1f} GiB) from {path} in {dt:.1f}s "
          f"({total / 2**30 / dt:.1f} GiB/s)", flush=True)


def register():
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.utils.torch_utils import set_default_torch_dtype

    @register_model_loader("k3snap")
    class K3SnapLoader(DefaultModelLoader):
        def __init__(self, load_config):
            import copy
            real = copy.copy(load_config)
            real.load_format = os.environ.get("K3SNAP_BASE_FORMAT", "fastsafetensors")
            super().__init__(real)

        def load_model(self, vllm_config, model_config, prefix=""):
            from vllm.distributed import get_tp_group

            path, key_src = _snapshot_dir(vllm_config)
            # Loading is collective (fastsafetensors shares files across ranks), so
            # every rank must take the same path: use the snapshot only if all have it.
            have = torch.tensor([1 if os.path.exists(os.path.join(path, "COMPLETE")) else 0],
                                device="cuda", dtype=torch.int32)
            torch.distributed.all_reduce(have, op=torch.distributed.ReduceOp.MIN,
                                         group=get_tp_group().device_group)
            if have.item() == 0:
                print(f"[k3snap] no snapshot at {path} ({key_src}); loading checkpoint", flush=True)
                model = super().load_model(vllm_config, model_config, prefix)
                torch.cuda.synchronize()
                if not os.path.exists(os.path.join(path, "COMPLETE")):
                    _save(model, path)
                return model
            load_device = vllm_config.load_config.device or vllm_config.device_config.device
            target_device = torch.device(load_device)
            with set_default_torch_dtype(model_config.dtype):
                with target_device:
                    model = self.create_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
                process_weights_after_loading(model, model_config, target_device)
                _load(model, path)
            return model.eval()

    # Keep the base class referenced so static checkers don't drop the import.
    assert issubclass(K3SnapLoader, BaseModelLoader)
