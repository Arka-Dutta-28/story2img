"""
generator/backends/base.py
---------------------------
Abstract base classes and data structures for image generation backends.

Responsibilities
----------------
- Define the GeneratedImage dataclass for metadata-rich image results
- Provide the BaseGeneratorBackend abstract interface for all backends
- Establish the contract that all backends must implement
- Support backend feature detection (ControlNet, img2img)

Architecture role
-----------------
This module serves as the foundation for the backend abstraction layer.
All concrete backends inherit from BaseGeneratorBackend and must implement
its abstract methods. The GeneratedImage dataclass ensures consistent
metadata across all backends while allowing backend-specific extensions.

Backend contract
----------------
Backends must implement:
- load_pipeline(): Initialize and cache the diffusion pipeline
- unload_pipeline(): Clean up GPU memory and cached models
- generate(): Produce images from prompts with reproducible seeds
- supports_controlnet(): Return True if ControlNet is available
- supports_img2img(): Return True if img2img is available (future use)

Supported features
------------------
- Unified pipeline lifecycle management
- Automatic device/dtype selection
- VRAM usage logging
- Seed-based reproducibility
- Backend-specific metadata in results

Unsupported features
--------------------
- Multi-GPU distribution (single device only)
- Dynamic pipeline reconfiguration (requires unload/reload)
- Concurrent generation (single-threaded per backend instance)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class GeneratedImage:
    """
    Container for generated image with comprehensive metadata.

    This dataclass holds the PIL image along with all generation parameters
    and backend-specific information needed for reproducibility and debugging.

    Parameters
    ----------
    image : Image.Image
        The generated PIL image.
    seed : int
        Random seed used for generation (ensures reproducibility).
    candidate_idx : int
        Index of this image in the batch (0-based).
    prompt : str
        Text prompt used for generation.
    steps : int
        Number of denoising steps.
    guidance_scale : float
        Classifier-free guidance scale.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    backend_name : str, optional
        Name of the backend that generated this image.
    precision : str, optional
        Precision mode used (fp16, fp32, fp8, etc.).
    quantized : bool, optional
        Whether quantization was used.
    generation_time : float, optional
        Time taken to generate this image in seconds.

    Notes
    -----
    All fields are immutable after creation. Backend-specific fields like
    precision and quantized allow tracking of generation characteristics
    for debugging and optimization.
    """
    image: Image.Image
    seed: int
    candidate_idx: int
    prompt: str
    steps: int
    guidance_scale: float
    width: int
    height: int
    backend_name: str = ""
    precision: str = ""
    quantized: bool = False
    generation_time: float = 0.0


class BaseGeneratorBackend(ABC):
    """
    Abstract base class for all image generation backends.

    This class defines the interface that all backends must implement.
    It provides common functionality like config management and feature
    detection while requiring backends to implement core generation logic.

    Parameters
    ----------
    generation_config : dict[str, Any]
        Backend-specific generation settings from config.
    full_config : dict[str, Any]
        Complete configuration dictionary for cross-backend access.

    Notes
    -----
    Backends should implement lazy loading - pipelines are only loaded
    when load_pipeline() is called. This allows backend switching without
    unnecessary memory usage. The pipeline attribute should be set to None
    initially and populated on load_pipeline().

    Architecture behavior
    ---------------------
    - Single backend instance per process (managed by backend_factory)
    - Config updates via update_config() method for parameter changes
    - Explicit lifecycle management prevents VRAM leaks
    - Feature flags guide orchestration layer decisions

    Raises
    ------
    NotImplementedError
        If abstract methods are not implemented by concrete backends.
    """

    backend_name: str = "base"

    def __init__(self, generation_config: dict[str, Any], full_config: dict[str, Any]) -> None:
        self.generation_config = generation_config
        self.full_config = full_config
        self.pipeline = None

    @abstractmethod
    def load_pipeline(self) -> Any:
        """
        Load and initialize the diffusion pipeline.

        This method should load the model weights, configure the pipeline
        with appropriate device/dtype settings, and cache the pipeline for
        reuse. Subsequent calls should return the cached pipeline if already
        loaded and config unchanged.

        Returns
        -------
        Any
            The initialized diffusion pipeline object.

        Notes
        -----
        Implementations should:
        - Check if pipeline is already loaded and config unchanged
        - Select appropriate device (CUDA > MPS > CPU)
        - Configure dtype based on precision settings
        - Enable optimizations like xformers if available
        - Log VRAM usage before/after loading

        Raises
        ------
        RuntimeError
            If pipeline loading fails due to missing dependencies or invalid config.
        ImportError
            If required libraries are not installed.
        """
        raise NotImplementedError

    @abstractmethod
    def unload_pipeline(self) -> None:
        """
        Clean up the diffusion pipeline and free GPU memory.

        This method should explicitly delete the pipeline, clear CUDA cache,
        and run garbage collection to prevent VRAM leaks when switching
        backends or shutting down.

        Notes
        -----
        Implementations should:
        - Delete the pipeline object
        - Call torch.cuda.empty_cache() if available
        - Run gc.collect() to clean up Python objects
        - Reset internal pipeline state to None
        - Log VRAM cleanup for monitoring

        Edge cases
        ----------
        Safe to call multiple times or when pipeline is None.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, seed: int, config: dict[str, Any]) -> Image.Image:
        """
        Generate a single image from the given prompt and seed.

        This is the core generation method that all backends must implement.
        It should use the loaded pipeline to generate one image with full
        reproducibility based on the seed.

        Parameters
        ----------
        prompt : str
            Text prompt for image generation.
        seed : int
            Random seed for reproducible generation.
        config : dict[str, Any]
            Generation parameters (steps, guidance_scale, dimensions, etc.).

        Returns
        -------
        Image.Image
            The generated PIL image.

        Notes
        -----
        Implementations should:
        - Set the random seed before generation
        - Extract parameters from config dict
        - Use the cached pipeline for efficiency
        - Return only the PIL image (metadata added by caller)

        Raises
        ------
        RuntimeError
            If pipeline is not loaded or generation fails.
        ValueError
            If parameters are invalid.
        """
        raise NotImplementedError

    def supports_controlnet(self) -> bool:
        """
        Check if this backend supports ControlNet conditioning.

        Returns
        -------
        bool
            True if ControlNet is supported, False otherwise.

        Notes
        -----
        Default implementation returns False. Backends that support
        ControlNet should override this method to return True.
        """
        return False

    def supports_img2img(self) -> bool:
        """
        Check if this backend supports image-to-image generation.

        Returns
        -------
        bool
            True if img2img is supported, False otherwise.

        Notes
        -----
        Default implementation returns False. This is reserved for
        future backends that support img2img workflows.
        """
        return False

    def generate_controlled(
        self,
        prompt: str,
        control_image: Image.Image,
        seed: int,
        config: dict[str, Any],
    ) -> Image.Image:
        raise NotImplementedError(
            f"{self.backend_name} does not support ControlNet generation."
        )

    def update_config(self, generation_config: dict[str, Any], full_config: dict[str, Any]) -> None:
        self.generation_config = generation_config
        self.full_config = full_config
