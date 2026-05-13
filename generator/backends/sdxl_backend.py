"""
generator/backends/sdxl_backend.py
----------------------------------
SDXL backend implementation with ControlNet support.

Responsibilities
----------------
- Load and manage SDXL diffusion pipelines
- Handle device/dtype selection (CUDA > MPS > CPU)
- Support ControlNet conditioning for layout control
- Provide VRAM usage monitoring and cleanup
- Enable xformers memory optimization when available

Architecture role
-----------------
This backend implements the BaseGeneratorBackend interface for SDXL models.
It maintains separate pipelines for standard and ControlNet generation to
avoid reloading overhead. Device selection prioritizes GPU acceleration
with fallback to CPU for compatibility.

Backend-specific behavior
-------------------------
- Uses fp16 on GPU, fp32 on CPU for optimal performance/compatibility
- Caches pipelines based on model_id, dtype, and device
- Supports ControlNet depth conditioning for layout control
- Enables xformers attention optimization when available
- Logs VRAM usage for monitoring and debugging

Caching/lifecycle behavior
--------------------------
- Pipelines cached until model_id, dtype, or device changes
- ControlNet pipelines cached by (base_model, controlnet_model) pair
- Explicit unload_pipeline() required for memory cleanup
- update_config() allows parameter changes without reload

Supported features
------------------
- Standard SDXL text-to-image generation
- ControlNet depth conditioning
- Automatic device/dtype selection
- xformers memory optimization
- VRAM monitoring and cleanup

Unsupported features
--------------------
- img2img generation (reserved for future)
- Multiple ControlNet models simultaneously
- Dynamic LoRA adapter loading
- Multi-GPU distribution
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Optional

from .base import BaseGeneratorBackend

logger = logging.getLogger(__name__)


class SDXLBackend(BaseGeneratorBackend):
    """
    SDXL backend with ControlNet support for layout-controlled generation.

    This backend provides SDXL generation with optional ControlNet conditioning.
    It automatically selects the best available device and dtype, caches pipelines
    for performance, and provides comprehensive VRAM monitoring.

    Parameters
    ----------
    generation_config : dict[str, Any]
        SDXL-specific generation settings.
    full_config : dict[str, Any]
        Complete configuration for cross-backend access.

    Notes
    -----
    Device selection order: CUDA (fp16) > MPS (fp16) > CPU (fp32).
    Pipelines are cached to avoid redundant loading. ControlNet requires
    separate pipeline instance to maintain standard generation performance.

    Architecture constraints
    ------------------------
    - Single pipeline instance per backend (standard or ControlNet)
    - Device/dtype changes require pipeline reload
    - ControlNet model changes invalidate cached ControlNet pipeline
    - update_config() allows parameter-only changes without reload
    """

    backend_name = "sdxl"

    def __init__(self, generation_config: dict[str, Any], full_config: dict[str, Any]) -> None:
        self.generation_config = generation_config
        self.full_config = full_config
        self.pipeline = None
        self._controlnet_pipeline = None
        self._loaded_model_id: Optional[str] = None
        self._loaded_dtype: Optional[str] = None
        self._device: Optional[str] = None
        self._controlnet_key: Optional[tuple[str, str]] = None

    def _log_vram(self, stage: str) -> None:
        """
        Log current CUDA VRAM usage for monitoring.

        Parameters
        ----------
        stage : str
            Descriptive stage name for the log message.

        Notes
        -----
        Only logs when CUDA is available. Silently ignores errors
        to avoid disrupting generation flow.
        """
        try:
            import torch

            if torch.cuda.is_available():
                logger.info(
                    "%s | CUDA memory: allocated=%d MiB reserved=%d MiB",
                    stage,
                    torch.cuda.memory_allocated() // (1024 * 1024),
                    torch.cuda.memory_reserved() // (1024 * 1024),
                )
        except Exception:
            pass

    def _select_device_dtype(self) -> tuple[str, Any]:
        """
        Select the best available device and corresponding dtype.

        Returns
        -------
        tuple[str, Any]
            (device_name, torch_dtype) pair.

        Notes
        -----
        Selection priority: CUDA (fp16) > MPS (fp16) > CPU (fp32).
        fp16 provides best performance on GPU, fp32 ensures compatibility on CPU.
        Logs warning for CPU fallback due to expected slow performance.

        Edge cases
        ----------
        Falls back gracefully when preferred devices unavailable.
        """
        import torch

        if torch.cuda.is_available():
            return "cuda", torch.float16
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        logger.warning(
            "No GPU detected — SDXL will run on CPU. Generation will be very slow. "
            "Consider using smaller resolutions and fewer steps."
        )
        return "cpu", torch.float32

    def _get_model_id(self) -> str:
        return str(
            self.generation_config.get(
                "model_id",
                self.full_config.get("backends", {}).get("sdxl", {}).get(
                    "model_id",
                    "stabilityai/stable-diffusion-xl-base-1.0",
                ),
            )
        )

    def load_pipeline(self) -> Any:
        """
        Load or reuse the SDXL diffusion pipeline.

        Returns the cached pipeline if already loaded with matching config,
        otherwise loads a new pipeline with current settings.

        Returns
        -------
        Any
            The initialized StableDiffusionXLPipeline.

        Notes
        -----
        Pipeline caching based on model_id, dtype, and device prevents
        redundant loading. Enables xformers optimization on GPU devices.
        Logs VRAM usage before/after loading for monitoring.

        Raises
        ------
        ImportError
            If required packages (diffusers, torch) are not installed.

        Edge cases
        ----------
        Reuses cached pipeline when config unchanged.
        Safetensors variant automatically selected for fp16.
        """
        generation_config = self.generation_config
        model_id = self._get_model_id()
        device, dtype = self._select_device_dtype()
        dtype_name = str(dtype).split(".")[-1]

        if (
            self.pipeline is not None
            and self._loaded_model_id == model_id
            and self._loaded_dtype == dtype_name
            and self._device == device
        ):
            logger.info("Reusing cached SDXL pipeline for %s", model_id)
            self._log_vram("SDXL pipeline reuse")
            return self.pipeline

        self.unload_pipeline()
        self._log_vram("SDXL before load")

        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except ImportError as exc:
            raise ImportError(
                "Required SDXL packages are not installed. Run:\n"
                "  pip install diffusers transformers accelerate torch"
            ) from exc

        logger.info("Loading SDXL pipeline: %s", model_id)
        logger.info("SDXL device=%s dtype=%s", device, dtype_name)

        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
        )
        pipe = pipe.to(device)

        if device != "cpu":
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory-efficient attention enabled")
            except Exception:
                logger.debug("xformers not available; skipping")

        self.pipeline = pipe
        self._loaded_model_id = model_id
        self._loaded_dtype = dtype_name
        self._device = device
        self._log_vram("SDXL after load")
        logger.info("SDXL pipeline ready")
        return self.pipeline

    def _load_controlnet_pipeline(self, controlnet_model_id: str) -> Any:
        try:
            import torch
            from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        except ImportError as exc:
            raise ImportError(
                "diffusers / torch required for SDXL ControlNet. "
                "Run: pip install diffusers transformers accelerate torch"
            ) from exc

        base_model_id = self._get_model_id()
        key = (base_model_id, controlnet_model_id)
        device, dtype = self._select_device_dtype()
        dtype_name = str(dtype).split(".")[-1]

        if self._controlnet_pipeline is not None and self._controlnet_key == key:
            logger.info(
                "Reusing cached SDXL ControlNet pipeline for %s + %s",
                base_model_id,
                controlnet_model_id,
            )
            self._log_vram("SDXL ControlNet reuse")
            return self._controlnet_pipeline

        self._log_vram("SDXL ControlNet before load")
        logger.info(
            "Loading SDXL ControlNet pipeline: base=%s controlnet=%s",
            base_model_id,
            controlnet_model_id,
        )

        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=dtype,
        )

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
        )
        pipe = pipe.to(device)

        if device != "cpu":
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                logger.debug("xformers not available for ControlNet pipeline")

        self._controlnet_pipeline = pipe
        self._controlnet_key = key
        self._log_vram("SDXL ControlNet after load")
        logger.info("SDXL ControlNet pipeline ready")
        return pipe

    def unload_pipeline(self) -> None:
        """
        Clean up all cached pipelines and free GPU memory.

        Deletes both standard and ControlNet pipelines, clears CUDA cache,
        and resets internal state. Safe to call multiple times.

        Notes
        -----
        Explicitly deletes pipeline objects and calls garbage collection
        to prevent VRAM leaks. Clears CUDA cache when available.
        Resets all cached state variables.
        """
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        if self._controlnet_pipeline is not None:
            del self._controlnet_pipeline
            self._controlnet_pipeline = None
            self._controlnet_key = None

        self._loaded_model_id = None
        self._loaded_dtype = None
        self._device = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache after SDXL unload")
        except Exception:
            pass

        gc.collect()

    def generate(self, prompt: str, seed: int, config: dict[str, Any]) -> Any:
        """
        Generate a single SDXL image from prompt and seed.

        Parameters
        ----------
        prompt : str
            Text prompt for generation.
        seed : int
            Random seed for reproducible results.
        config : dict[str, Any]
            Generation parameters (steps, guidance_scale, dimensions).

        Returns
        -------
        Any
            PIL Image object of the generated image.

        Notes
        -----
        Uses the standard SDXL pipeline (not ControlNet). Automatically
        loads pipeline if not cached. Parameters extracted from config
        with sensible defaults.

        Raises
        ------
        ImportError
            If torch is not available.
        RuntimeError
            If pipeline loading or generation fails.
        """
        pipe = self.load_pipeline()

        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for SDXL generation.") from exc

        generator = torch.Generator(device=pipe.device.type).manual_seed(seed)
        steps = int(self.generation_config.get("steps", 30))
        guidance_scale = float(self.generation_config.get("guidance_scale", 7.5))
        height = int(self.generation_config.get("height", 768))
        width = int(self.generation_config.get("width", 768))

        output = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            num_images_per_prompt=1,
        )
        return output.images[0]

    def supports_controlnet(self) -> bool:
        return True

    def generate_controlled(
        self,
        prompt: str,
        control_image: Any,
        seed: int,
        config: dict[str, Any],
    ) -> Any:
        """
        Generate SDXL image with ControlNet conditioning.

        Parameters
        ----------
        prompt : str
            Text prompt for generation.
        control_image : Any
            PIL Image for ControlNet conditioning.
        seed : int
            Random seed for reproducible results.
        config : dict[str, Any]
            Generation parameters including controlnet settings.

        Returns
        -------
        Any
            PIL Image object of the generated image.

        Notes
        -----
        Uses ControlNet depth conditioning by default. ControlNet model
        can be configured via controlnet.model_id. Automatically loads
        ControlNet pipeline if needed.

        Raises
        ------
        NotImplementedError
            If ControlNet not supported (should not happen for SDXL).
        ImportError
            If torch not available.
        RuntimeError
            If pipeline loading or generation fails.
        """
        if not self.supports_controlnet():
            raise NotImplementedError("This backend does not support ControlNet.")

        cn_block = config.get("controlnet", {})
        controlnet_model_id = str(
            cn_block.get(
                "model_id",
                self.full_config.get("backends", {}).get("sdxl", {}).get(
                    "controlnet_model_id",
                    "diffusers/controlnet-depth-sdxl-1.0",
                ),
            )
        )

        pipe = self._load_controlnet_pipeline(controlnet_model_id)

        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for SDXL ControlNet generation.") from exc

        generator = torch.Generator(device=pipe.device.type).manual_seed(seed)
        steps = int(self.generation_config.get("steps", 30))
        guidance_scale = float(self.generation_config.get("guidance_scale", 7.5))
        height = int(self.generation_config.get("height", 768))
        width = int(self.generation_config.get("width", 768))
        conditioning_scale = float(cn_block.get("conditioning_scale", 0.5))

        output = pipe(
            prompt=prompt,
            image=control_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            controlnet_conditioning_scale=conditioning_scale,
            num_images_per_prompt=1,
        )
        return output.images[0]

    def update_config(self, generation_config: dict[str, Any], full_config: dict[str, Any]) -> None:
        self.generation_config = generation_config
        self.full_config = full_config
