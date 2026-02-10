from __future__ import annotations

import json
import platform

import torch


def main() -> None:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info.update(
            {
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            }
        )
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
