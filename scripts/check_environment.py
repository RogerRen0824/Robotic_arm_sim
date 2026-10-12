"""Read-only diagnostics. Does not install packages or alter environments."""
import importlib
import json
import platform
import sys

report = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
for name in ("numpy", "torch", "cv2", "mujoco", "robosuite"):
    try:
        module = importlib.import_module(name)
        item = {"available": True, "version": getattr(module, "__version__", "unknown")}
        if name == "torch":
            item["cuda_available"] = module.cuda.is_available()
            if item["cuda_available"]:
                item["gpu"] = module.cuda.get_device_name(0)
        report[name] = item
    except Exception as exc:
        report[name] = {"available": False, "error": str(exc)}
print(json.dumps(report, indent=2, ensure_ascii=False))
