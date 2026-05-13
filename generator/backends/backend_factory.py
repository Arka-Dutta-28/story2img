"""
generator/backends/backend_factory.py
-------------------------------------
Factory for backend selection and lifecycle management.

Responsibilities
----------------
- Provide unified backend instantiation based on config
- Manage global backend state and prevent memory leaks
- Handle backend switching with proper cleanup
- Cache backends to avoid redundant initialization

Architecture role
-----------------
This module implements the factory pattern for backend creation and
acts as the single point of access for backend instances. It maintains
a global registry of available backends and ensures only one backend
is active at a time to prevent VRAM conflicts.

Backend lifecycle management
----------------------------
- Backends are cached globally to avoid reload overhead
- Config updates are handled via update_config() method
- Switching backends triggers unload_pipeline() on the old backend
- Explicit unload_current_backend() for cleanup on shutdown

Caching behavior
----------------
- Same backend type reuses cached instance with config updates
- Different backend type triggers unload/load cycle
- No caching across different backend classes

Supported backends
------------------
- sdxl: SDXL backend with ControlNet support
- flux: FLUX.1-dev backend with quantization options

Edge cases
----------
- Invalid backend names raise ValueError with supported options
- Config without 'generation.backend' defaults to 'sdxl'
- Missing backends section handled gracefully
"""

from __future__ import annotations

from typing import Any

from .base import BaseGeneratorBackend
from .flux_backend import FluxBackend
from .sdxl_backend import SDXLBackend

_BACKEND_REGISTRY: dict[str, type[BaseGeneratorBackend]] = {
    "sdxl": SDXLBackend,
    "flux": FluxBackend,
}

_current_backend: BaseGeneratorBackend | None = None


def _normalize_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Extract generation config from full config dict.

    Parameters
    ----------
    config : dict[str, Any]
        Input config that may contain 'generation' section.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        (generation_config, full_config) where generation_config
        is the 'generation' section or the whole config if missing.

    Notes
    -----
    This helper normalizes config access for backends that expect
    generation-specific settings vs full config for cross-backend access.
    """
    if "generation" in config:
        return config["generation"], config
    return config, {"generation": config}


def get_backend(config: dict[str, Any]) -> BaseGeneratorBackend:
    """
    Get or create the appropriate backend instance based on config.

    This function implements the factory pattern with caching. It selects
    the backend based on config["generation"]["backend"] and returns a
    cached instance if available, or creates a new one if needed.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary containing backend selection and settings.

    Returns
    -------
    BaseGeneratorBackend
        The selected backend instance, ready for pipeline loading.

    Notes
    -----
    Backend selection logic:
    - Extract backend name from config["generation"]["backend"]
    - Default to "sdxl" if not specified
    - Validate against _BACKEND_REGISTRY
    - Return cached instance if same backend type
    - Switch backends with proper cleanup if different

    Raises
    ------
    ValueError
        If specified backend name is not in the registry.

    Edge cases
    ----------
    Config without 'generation' section uses whole config as generation config.
    Backend names are case-insensitive (converted to lowercase).
    """
    global _current_backend
    generation_config, full_config = _normalize_config(config)
    backend_name = str(generation_config.get("backend", "sdxl")).lower()

    backend_cls = _BACKEND_REGISTRY.get(backend_name)
    if backend_cls is None:
        raise ValueError(
            f"Unknown generation backend: {backend_name!r}. "
            f"Supported backends: {', '.join(sorted(_BACKEND_REGISTRY))}."
        )

    if _current_backend is not None and _current_backend.backend_name == backend_name:
        _current_backend.update_config(generation_config, full_config)
        return _current_backend

    if _current_backend is not None:
        _current_backend.unload_pipeline()
        _current_backend = None

    _current_backend = backend_cls(generation_config, full_config)
    return _current_backend


def unload_current_backend() -> None:
    """
    Clean up the currently active backend and free its resources.

    This function should be called on application shutdown or when
    explicitly switching away from backends to prevent VRAM leaks.

    Notes
    -----
    Safe to call multiple times or when no backend is active.
    Calls unload_pipeline() on the current backend if present.
    Resets the global _current_backend to None.
    """
    global _current_backend
    if _current_backend is not None:
        _current_backend.unload_pipeline()
        _current_backend = None
        _current_backend = None
