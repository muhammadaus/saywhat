"""Import the plugin directory as the ``saywhat`` package, whatever the clone is named."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

if "saywhat" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "saywhat", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["saywhat"] = module
    spec.loader.exec_module(module)


class MemoryStore:
    """Stand-in for Hermes's ``ctx.state`` (get/set JSON values)."""

    def __init__(self):
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        import json
        self.data[key] = json.loads(json.dumps(value))  # enforce JSON-serializable like PluginState


@pytest.fixture
def store():
    return MemoryStore()
