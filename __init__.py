import importlib.util
import os
import sys

_DIR = os.path.dirname(os.path.realpath(__file__))
# ComfyUI already has a top-level module named nodes. Loading by file path
# keeps this pack from binding to that module and exporting no camera nodes.
_spec = importlib.util.spec_from_file_location(
    "camerathief_nodes",
    os.path.join(_DIR, "nodes.py"),
)
_nodes = importlib.util.module_from_spec(_spec)
sys.modules["camerathief_nodes"] = _nodes
_spec.loader.exec_module(_nodes)

NODE_CLASS_MAPPINGS = _nodes.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _nodes.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
