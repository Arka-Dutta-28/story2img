"""
generator/backends/flux_backend.py
----------------------------------
FLUX.1-dev backend implementation with precision and quantization support.

Responsibilities
----------------
- Load FLUX.1-dev diffusion pipelines with configurable precision
- Handle fp16, fp32, and fp8 quantized inference modes
- Manage device selection and memory optimization
- Provide VRAM usage monitoring and cleanup
- Support automatic fallback for unsupported configurations

Architecture role
-----------------
This backend implements the BaseGeneratorBackend interface for FLUX models.
It provides advanced precision options including 8-bit quantization for memory
efficiency, while maintaining compatibility with the unified backend interface.

Backend-specific behavior
-------------------------
- Supports fp16/fp32/fp8 precision modes with automatic selection
- Uses bitsandbytes for fp8 quantization (CUDA only)
- Falls back to fp32 on CPU when fp16 unavailable
- Validates precision configurations at load time
- Logs detailed VRAM usage for monitoring

Caching/lifecycle behavior
--------------------------
- Pipelines cached by model_id, precision, and quantization state
- Explicit unload_pipeline() required for memory cleanup
- Quantization changes require full pipeline reload
- update_config() allows parameter changes without reload

Supported features
------------------
- FLUX.1-dev text-to-image generation
- Multiple precision modes (fp16, fp32, fp8)
- Automatic device/dtype selection
- bitsandbytes quantization support
- VRAM monitoring and cleanup

Unsupported features
--------------------
- ControlNet conditioning (FLUX ControlNet not yet available)
- img2img generation (reserved for future)
- Multi-GPU distribution
- Dynamic LoRA adapter loading
- MPS acceleration (limited FLUX support)
"""

from __future__ import annotations

import gc
import logging
from typing import Any

from .base import BaseGeneratorBackend

logger = logging.getLogger(__name__)


class FluxBackend(BaseGeneratorBackend):
    """
    FLUX.1-dev backend with configurable precision and quantization.

    This backend provides FLUX generation with advanced memory optimization
    options including 8-bit quantization. It automatically handles device
    selection and precision fallbacks for optimal performance.

    Parameters
    ----------
    generation_config : dict[str, Any]
        FLUX-specific generation settings including precision.
    full_config : dict[str, Any]
        Complete configuration for cross-backend access.

    Notes
    -----
    Supports fp16/fp32/fp8 precision modes. fp8 automatically enables
    quantization using bitsandbytes. Precision validation occurs at
    pipeline load time with clear error messages.

    Architecture constraints
    ------------------------
    - Quantization requires CUDA and bitsandbytes
    - fp8 precision forces quantization=True
    - CPU fallback uses fp32 regardless of precision setting
    - Pipeline reload required for precision/quantization changes
    - update_config() allows parameter-only changes without reload

    Raises
    ------
    ValueError
        If unsupported precision mode specified.
    RuntimeError
        If quantized mode requested without CUDA/bitsandbytes.
    """

    backend_name = "flux"

    def __init__(self, generation_config: dict[str, Any], full_config: dict[str, Any]) -> None:
        self.generation_config = generation_config
        self.full_config = full_config
        self.pipeline = None
        self._loaded_model_id: str | None = None
        self._loaded_precision: str | None = None
        self._quantized: bool = False
        self._device: str | None = None

    def _log_vram(self, stage: str) -> None:
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

    def _select_device(self) -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        logger.warning(
            "No GPU detected — FLUX will run on CPU. Generation will be extremely slow."
        )
        return "cpu"

    def _get_model_id(self) -> str:
        return str(
            self.generation_config.get(
                "model_id",
                self.full_config.get("backends", {}).get("flux", {}).get(
                    "model_id",
                    "black-forest-labs/FLUX.1-dev",
                ),
            )
        )

    def _get_precision(self) -> tuple[str, bool]:
        """
        Determine precision mode and quantization setting from config.

        Returns
        -------
        tuple[str, bool]
            (precision_mode, quantized_flag) pair.

        Notes
        -----
        fp8 precision automatically enables quantization. Validates
        precision against supported modes. Falls back to config defaults.

        Raises
        ------
        ValueError
            If precision mode is not fp16, fp32, or fp8.
        """
        precision = str(
            self.generation_config.get(
                "precision",
                self.full_config.get("backends", {}).get("flux", {}).get(
                    "precision",
                    "fp16",
                ),
            )
        ).lower()
        quantized = bool(
            self.generation_config.get(
                "quantized",
                self.full_config.get("backends", {}).get("flux", {}).get(
                    "quantized",
                    False,
                ),
            )
        )
        if precision == "fp8":
            quantized = True
        if precision not in {"fp16", "fp32", "fp8"}:
            raise ValueError(
                f"Unsupported FLUX precision: {precision!r}. "
                "Supported values: fp16, fp32, fp8."
            )
        return precision, quantized

    def load_pipeline(self) -> Any:
        """
        Load or reuse the FLUX diffusion pipeline with specified precision.

        Returns the cached pipeline if already loaded with matching config,
        otherwise loads a new pipeline with current precision settings.

        Returns
        -------
        Any
            The initialized FLUX DiffusionPipeline.

        Notes
        -----
        Handles precision-specific loading: fp8 uses quantization,
        fp16/fp32 use standard torch dtypes. Validates CUDA requirement
        for quantization. Logs detailed loading information.

        Raises
        ------
        RuntimeError
            If quantization requested without CUDA/bitsandbytes support.
        ImportError
            If required packages not available.

        Edge cases
        ----------
        CPU fallback forces fp32 regardless of precision setting.
        fp8 precision automatically enables quantization.
        """
        import torch

        from diffusers import DiffusionPipeline

        model_id = self._get_model_id()
        precision, quantized = self._get_precision()
        device = self._select_device()
        self._device = device

        if (
            self.pipeline is not None
            and self._loaded_model_id == model_id
            and self._loaded_precision == precision
            and self._quantized == quantized
        ):
            logger.info("Reusing cached FLUX pipeline for %s", model_id)
            self._log_vram("FLUX pipeline reuse")
            return self.pipeline

        self._loaded_precision = precision
        self._quantized = quantized

        self.unload_pipeline()
        self._log_vram("FLUX before load")

        if quantized and device != "cuda":
            raise RuntimeError(
                "FLUX quantized inference requires CUDA and bitsandbytes support."
            )

        logger.info(
            "Loading FLUX pipeline: %s | precision=%s | quantized=%s | device=%s",
            model_id,
            precision,
            quantized,
            device,
        )

        if quantized:
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    load_in_8bit=True,
                    device_map="auto",
                )
            except Exception as exc:
                raise RuntimeError(
                    "Unable to load FLUX model in 8-bit quantized mode. "
                    "Ensure bitsandbytes is installed and CUDA is available."
                ) from exc
        else:
            dtype = torch.float16 if precision == "fp16" else torch.float32
            if device == "cpu" and precision == "fp16":
                logger.warning(
                    "CUDA not available — falling back to CPU with fp32 for FLUX."
                )
                dtype = torch.float32
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
            )
            pipe = pipe.to(device)

        self.pipeline = pipe
        self._loaded_model_id = model_id
        self._log_vram("FLUX after load")
        logger.info("FLUX pipeline ready")
        return self.pipeline

    def unload_pipeline(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        self._loaded_model_id = None
        self._loaded_precision = None
        self._quantized = False
        self._device = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache after FLUX unload")
        except Exception:
            pass

        gc.collect()

    def generate(self, prompt: str, seed: int, config: dict[str, Any]) -> Any:
        """
        Generate a single FLUX image from prompt and seed.

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
        Uses the configured FLUX pipeline with precision optimizations.
        Automatically loads pipeline if not cached. Parameters extracted
        from config with sensible defaults.

        Raises
        ------
        ImportError
            If torch is not available.
        RuntimeError
            If pipeline loading or generation fails.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for FLUX generation.") from exc

        pipe = self.load_pipeline()
        generator = torch.Generator(device=self._device if self._device else "cpu").manual_seed(seed)
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
        """
        Confirm FLUX backend does not support ControlNet conditioning.

        Returns
        -------
        bool
            Always False for FLUX backend.

        Notes
        -----
        FLUX ControlNet models are not yet available. This may change
        in future versions as the FLUX ecosystem develops.
        """
        return False

    def update_config(self, generation_config: dict[str, Any], full_config: dict[str, Any]) -> None:
        self.generation_config = generation_config
        self.full_config = full_config
