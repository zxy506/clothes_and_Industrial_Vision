"""Record the active project's Python, package and CUDA environment."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    import torch
    import torchvision

    gpu = None
    if torch.cuda.is_available():
        a = torch.tensor([2.0], device="cuda")
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "tensor_test": float((a * 3).cpu().item()),
            "memory_allocated_bytes": torch.cuda.memory_allocated(0),
        }
    usage = shutil.disk_usage(Path.cwd())
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_logical_count": __import__("os").cpu_count(),
        "disk_free_bytes": usage.free,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
    }
    output = Path("logs/environment.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
