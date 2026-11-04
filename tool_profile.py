"""Select the requested tool version without launching Vitis or relaxing checks."""
import os

from scripts.verify_hls import validate_tool_version


DEFAULT_TOOL_VERSION = '2025.2'


def resolve_tool_version(explicit=None):
    """An explicit option overrides the environment, then the project default."""
    version = explicit if explicit is not None else os.environ.get('HLS_TOOL_VERSION', DEFAULT_TOOL_VERSION)
    validate_tool_version(version)
    return version
