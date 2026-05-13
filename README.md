# story2img

A Python repository for automated story-to-image generation. Converts text narratives into illustrated sequences using multiple diffusion backends (SDXL, FLUX.1-dev) with optional layout-controlled generation via ControlNet. Integrates LLM-based story parsing, constraint-driven prompt building, OpenCLIP-based image evaluation, and per-character memory management.

## Core Design

The system follows a **modular orchestration architecture**:

```
Story text
   ↓
LLM Parser (Gemini/Groq/NVIDIA)  → Structured scenes (characters, descriptions)
   ↓
Constraint Builder                  → Validated prompts, optional layout
   ↓
Image Generator (SDXL/FLUX)        → N candidate images per scene
   ↓
OpenCLIP Evaluator                 → Best image selection via text/image similarity
   ↓
Memory Manager (v2 pipeline)        → Persist character reference images
```

Each stage operates independently; the pipeline can be run in full or invoked selectively (e.g., **parse only**, **generate only**, **evaluate only**).

## Features

- **Multi-backend image generation** — Clean backend abstraction layer supporting SDXL and FLUX.1-dev; simple to extend with new models
- **SDXL support** — Text-to-image generation with optional ControlNet depth conditioning for layout control
- **FLUX.1-dev support** — Latest diffusion model with configurable precision (fp16, fp32, fp8 quantized)
- **Config-driven backend selection** — Switch between SDXL and FLUX via `config.yaml` without code changes
- **Optional ControlNet** — SDXL-exclusive layout-derived depth conditioning for spatial control over character placement
- **Deterministic seed generation** — Incremental or random seed strategies for reproducibility
- **Candidate-based image selection** — Generate N images per scene, select best via OpenCLIP scoring
- **Backend lifecycle management** — Explicit pipeline loading/unloading with VRAM cleanup; prevents memory leaks in multi-model systems
- **Quantized inference** — fp8 quantization support (CUDA + bitsandbytes) for reduced VRAM footprint
- **Per-character memory (v2 pipeline)** — Track reference images across scenes for consistent character appearance
- **LLM-based story parsing** — Validate and structure raw story text into scenes with characters and style
- **Constraint-driven prompts** — Combine story details with style guides and optional Groq-based layout planning
- **OpenCLIP evaluation** — Weighted scoring across scene alignment, character identity, and temporal coherence
- **Text embedding cache** — Optional on-disk caching of CLIP embeddings for repeated evaluations

## Architecture Overview

### Project Structure

```
generator/
├── backends/                   # Backend abstraction layer
│   ├── base.py               # Abstract BaseGeneratorBackend interface
│   ├── backend_factory.py     # Factory for backend selection & lifecycle
│   ├── sdxl_backend.py        # SDXL implementation with ControlNet
│   └── flux_backend.py        # FLUX implementation with precision modes
├── image_generator.py          # Orchestration: backend selection → metadata assembly
└── controlled_generator.py     # ControlNet path: layout → control image → backend

llm/
├── base.py                     # LLMBase, LLMResponse interfaces
├── parser.py                   # Story parser with validation
├── gemini_client.py           # Gemini provider
├── groq_client.py             # Groq provider
├── nvidia_client.py           # NVIDIA OpenAI-compatible provider
└── __init__.py                 # Client factory

pipeline/
├── story_pipeline.py           # Phase 5: Basic generation + evaluation (no memory)
└── story_pipeline_v2.py        # Phase 7: Memory + constraints + optional ControlNet

evaluator/
└── clip_evaluator.py          # OpenCLIP text-image and image-image scoring

constraints/
└── constraint_builder.py       # Structured constraints, prompt assembly, optional Groq layout

memory/
└── memory_manager.py          # Per-character reference image storage

utils/
└── sanity_check_*.py          # Development/testing scripts (parser, generator, evaluator, etc.)
```

### Backend Abstraction Layer

All image generation flows through the **backend abstraction**:

1. **Backend Selection** (`backend_factory.py`)
   - Reads `generation.backend` from config
   - Instantiates appropriate backend class (SDXL or FLUX)
   - Manages global backend instance with explicit lifecycle

2. **Backend Interface** (`base.py`)
   - Abstract `BaseGeneratorBackend` class
   - Required methods: `load_pipeline()`, `unload_pipeline()`, `generate()`
   - Feature flags: `supports_controlnet()`, `supports_img2img()`
   - Unified `GeneratedImage` dataclass for metadata across all backends

3. **Orchestration** (`image_generator.py`, `controlled_generator.py`)
   - Fully backend-agnostic
   - Delegates generation to selected backend
   - Assembles consistent metadata (seed, prompt, generation_time, backend_name, precision, quantized)

### Image Generation Pipeline

**Standard path:**
```python
prompt → backend.load_pipeline() → backend.generate(prompt, seed) → Image
```

**ControlNet path (SDXL only):**
```python
prompt + layout → layout_to_control_image() → backend.generate_controlled() → Image
```

## Backend System

### SDXL Backend

**Supported features:**
- SDXL text-to-image generation
- ControlNet depth conditioning (layout-derived)
- Device auto-selection: CUDA (fp16) > MPS (fp16) > CPU (fp32)
- xformers memory optimization on GPU
- Pipeline caching by model_id/dtype/device
- Separate ControlNet pipeline to avoid reloading overhead

**VRAM estimates (1024×1024):**
- Standard generation: ~14–16 GB (CUDA, fp16)
- With ControlNet: ~18–20 GB (additional ControlNet pipeline)
- CPU fallback: Works but extremely slow; not recommended

**Limitations:**
- CPU inference impractical for real-time use
- Largest improvements from xformers (GPU only)
- ControlNet always uses depth model (no alternative types)

### FLUX Backend

**Supported features:**
- FLUX.1-dev text-to-image generation
- Precision modes: fp16, fp32, fp8 quantized
- Device auto-selection: CUDA > MPS > CPU
- 8-bit quantization support (CUDA + bitsandbytes)
- Pipeline caching by model_id/precision/quantization state

**Precision modes:**

| Mode | Memory Impact | Speed | Hardware Support | Best For |
|------|---------------|-------|------------------|----------|
| fp32 | ~24 GB | Baseline | All | CPU, high precision |
| fp16 | ~14 GB | ~2× faster | CUDA, MPS | GPU, balanced |
| fp8 (quantized) | ~8 GB | ~2× faster | CUDA only | High-VRAM-constrained systems |

**VRAM estimates (1024×1024):**
- fp32: ~24 GB
- fp16: ~14 GB
- fp8: ~8 GB (with bitsandbytes)

**Limitations:**
- ControlNet not yet supported (FLUX ControlNet models may not exist yet)
- fp8 requires CUDA and bitsandbytes
- CPU fallback uses fp32 regardless of precision setting
- MPS support is limited; CUDA strongly recommended

### Backend Comparison

| Feature | SDXL | FLUX |
|---------|------|------|
| **ControlNet support** | Yes (depth) | No |
| **Quantization support** | No | Yes (fp8) |
| **Min VRAM (1024×1024)** | ~14 GB (fp16) | ~8 GB (fp8) |
| **Precision modes** | fp16 (GPU), fp32 (CPU) | fp16, fp32, fp8 |
| **Speed** | Baseline | Similar (model dependent) |
| **Maturity** | Stable | Active development |

## Installation

### Requirements

- **Python:** 3.10+
- **CUDA:** Optional but strongly recommended for GPU inference
  - CUDA 11.8+ for SDXL/FLUX
  - bitsandbytes required for FLUX fp8 quantization
- **GPU:** Recommended 16+ GB VRAM for comfortable use at 1024×1024
  - SDXL alone: 14–16 GB
  - SDXL + ControlNet: 18–20 GB
  - FLUX fp16: 14 GB
  - FLUX fp8: 8 GB

### Setup

1. **Clone and navigate:**
   ```bash
   cd /path/to/story2img
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note:** `requirements.txt` is a complete environment lockfile. If installation fails:
   - Check that your CUDA drivers match PyTorch CUDA version
   - Some entries may reference local paths; replace with PyPI pins if needed
   - Try `pip install --no-cache-dir` to avoid stale wheels

4. **Configure `config.yaml`:**
   - Copy or edit existing `config.yaml` at repo root
   - Set backend selection: `generation.backend: sdxl` or `generation.backend: flux`
   - Configure backend-specific settings under `backends.sdxl` or `backends.flux`
   - Set API keys for remote LLM providers (see API Keys section)

5. **Set API keys (for LLM features):**
   ```bash
   export GEMINI_API_KEY="your_key_here"     # Gemini
   export GROQ_API_KEY="your_key_here"       # Groq
   export NVIDIA_API_KEY="your_key_here"     # NVIDIA OpenAI-compatible
   ```

### Optional: GPU Acceleration

**For CUDA (recommended):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers accelerate xformers
```

**For FLUX fp8 quantization (CUDA only):**
```bash
pip install bitsandbytes
```

**For Apple Silicon (MPS, limited support):**
- Install standard PyTorch; MPS support is automatic
- Note: FLUX on MPS may have limited optimization; CUDA strongly preferred

## Configuration

All runtime behavior is controlled via `config.yaml` at the repo root.

### Generation Settings

```yaml
generation:
  backend: sdxl              # "sdxl" or "flux"
  num_candidates: 3          # Images to generate per scene
  steps: 30                  # Denoising steps (higher = slower + possibly better)
  guidance_scale: 7.5        # Classifier-free guidance strength
  height: 1024               # Output image height (pixels)
  width: 1024                # Output image width (pixels)
  seed_strategy: "incremental"  # "incremental" or "random"
  base_seed: 42              # Starting seed for incremental strategy
```

**Seed strategies:**
- `incremental`: Seeds are base_seed, base_seed+1, base_seed+2, etc.
  - Reproducible across runs with same base_seed
  - Useful for systematic variation
- `random`: Seeds are random within uint32 range
  - Non-reproducible; useful for diversity

### Backend-Specific Settings

#### SDXL Backend

```yaml
backends:
  sdxl:
    model_id: "stabilityai/stable-diffusion-xl-base-1.0"
    precision: "fp16"  # Only option; fp16 on GPU, fp32 on CPU
```

#### FLUX Backend

```yaml
backends:
  flux:
    model_id: "black-forest-labs/FLUX.1-dev"
    precision: "fp16"     # "fp16", "fp32", or "fp8"
    quantized: false      # true for fp8 quantization (CUDA only)
    load_in_8bit: true    # Use bitsandbytes for quantization
```

### ControlNet (SDXL only)

```yaml
controlnet:
  enabled: false                                      # true to enable
  model_id: "diffusers/controlnet-depth-sdxl-1.0"   # Depth ControlNet
  conditioning_scale: 0.5                             # 0.0–1.0, controls strength
  save_debug_control: true                            # Save control images for debugging
```

**Important:** ControlNet requires a valid layout in the constraint builder output. If layout generation fails, ControlNet generation is skipped with a warning.

### Other Configuration Sections

- **llm:** Parser provider selection and model settings (Gemini, Groq, NVIDIA)
- **evaluator:** OpenCLIP model and scoring weights
- **memory:** Per-character reference image tracking (v2 pipeline)
- **constraint_builder:** Style prefix/suffix for prompt assembly
- **layout:** Optional Groq-based layout planning
- **prompt_cache:** Enable prompt reuse per scene
- **logging:** Output directory and what to save

## Usage

### A. Code Understanding & Development

This is the **primary path** if you want to:
- Understand how the system works
- Modify or extend the code
- Run individual components

**Start with:**

1. **Image Generation** (`generator/image_generator.py`):
   ```python
   from generator import generate_images
   import yaml

   with open("config.yaml") as f:
       config = yaml.safe_load(f)

   # Generate 3 images
   images = generate_images(
       prompt="A lone traveler on a cliff at dusk",
       n=3,
       config=config
   )

   for idx, img in enumerate(images):
       print(f"Image {idx}: backend={img.backend_name}, precision={img.precision}")
       img.image.save(f"output_{idx}.png")
   ```

2. **Controlled Generation** (SDXL + ControlNet, `generator/controlled_generator.py`):
   ```python
   from generator.controlled_generator import generate_images_controlled

   layout = {
       "characters": [
           {"name": "hero", "position": "center", "depth": "midground"}
       ]
   }

   images = generate_images_controlled(
       prompt="A hero in an epic battle",
       layout=layout,
       n=2,
       config=config
   )
   ```

3. **Story Parsing** (`llm/parser.py`):
   ```python
   from llm import build_llm_client
   from llm.parser import StoryParser

   llm_client = build_llm_client(config["llm"])
   parser = StoryParser(llm_client)
   
   result = parser.parse("Once upon a time, a hero...")
   print(f"Characters: {[c['name'] for c in result['characters']]}")
   print(f"Scenes: {len(result['scenes'])}")
   ```

4. **Full Pipeline (v2 with memory)** (`pipeline/story_pipeline_v2.py`):
   ```python
   from pipeline.story_pipeline_v2 import run_pipeline_v2

   images_per_scene, memory_state = run_pipeline_v2(
       story_text="Story content here...",
       config=config,
       output_dir="story_output"
   )
   ```

5. **Development scripts** (`utils/sanity_check_*.py`):
   - **Parser test:** `python utils/sanity_check_parser.py` (LLM-based; requires API key)
   - **Backend test:** `python utils/sanity_check_generator_backends.py --backend flux --count 2`
   - **Full pipeline:** `python utils/sanity_check_pipeline_v2.py` (end-to-end)

### B. Execution (Full Pipeline)

**Only attempt if you have:**
- Adequate GPU resources (16+ GB VRAM recommended)
- API keys configured for LLM provider
- Realistic expectations about runtime (minutes to hours depending on scene count)

**Steps:**

1. Edit `config.yaml` with your settings (backend, LLM provider, output directory, etc.)
2. Set environment variables for API keys
3. Run from repo root:
   ```bash
   python utils/sanity_check_pipeline_v2.py
   # or import and call directly
   from pipeline.story_pipeline_v2 import run_pipeline_v2
   ```

**Expect:**
- Parser: ~30 seconds (1 API call)
- Image generation: 2–5 minutes per scene (depends on num_candidates, resolution, backend)
- Evaluation: ~10 seconds per batch
- Memory updates: Negligible

### Backend-Specific Testing

**SDXL smoke test:**
```bash
python utils/sanity_check_generator_backends.py --backend sdxl --count 2
```

**FLUX smoke test (fp16):**
```bash
python utils/sanity_check_generator_backends.py --backend flux --count 1
```

**FLUX quantized (fp8, CUDA only):**
Edit `config.yaml`:
```yaml
backends:
  flux:
    precision: "fp8"
    quantized: true
```
Then run the same command.

## Precision Modes & Quantization

### Understanding Precision

- **fp32 (float32):** Full 32-bit precision
  - Slowest, highest VRAM (~24 GB for FLUX at 1024×1024)
  - Best for CPU inference or maximum accuracy
  - FLUX default if no GPU detected

- **fp16 (float16):** Half precision
  - ~2× faster than fp32, ~40% VRAM savings
  - Minimal quality loss
  - Recommended for GPU inference
  - Supported by SDXL and FLUX

- **fp8 (8-bit quantization):** FLUX only
  - Extreme VRAM reduction (~8 GB for FLUX)
  - Potential quality degradation (minor in practice)
  - Requires CUDA + bitsandbytes
  - Useful for multi-model systems or memory-constrained setups

### When to Use Each Mode

| Scenario | Mode | Reason |
|----------|------|--------|
| 24GB VRAM, single FLUX generation | fp32 | Best precision, acceptable speed |
| 16GB VRAM, single SDXL or FLUX | fp16 | Balance of speed/precision/VRAM |
| 12GB VRAM, multiple models or ControlNet | fp8 (FLUX) | Extreme VRAM savings for flexibility |
| CPU-only | fp32 | fp16 unsupported on CPU |
| Apple Silicon | fp16 | MPS-accelerated; fp8 not applicable |

### FLUX fp8 Example

In `config.yaml`:
```yaml
generation:
  backend: flux

backends:
  flux:
    model_id: "black-forest-labs/FLUX.1-dev"
    precision: "fp8"
    quantized: true
    load_in_8bit: true
```

Then run generation as normal. On first load, bitsandbytes will quantize the model to 8-bit (one-time cost); subsequent loads reuse the quantized weights.

**Requirements:**
- CUDA available
- bitsandbytes installed
- ~8 GB VRAM

## Memory Management

### Lazy Loading

Pipelines are **not** loaded until `load_pipeline()` is called. This allows:
- Backend switching without unnecessary model loads
- Deferred GPU allocation until generation time
- Lower memory overhead during initialization

### Backend Lifecycle

```python
from generator.backends.backend_factory import get_backend, unload_current_backend

config = {"generation": {"backend": "sdxl"}, "backends": {...}}

# First call: loads SDXL pipeline
backend = get_backend(config)

# Subsequent calls with same backend: reuses cached pipeline
backend = get_backend(config)

# Switch to different backend: unloads SDXL, loads FLUX
config["generation"]["backend"] = "flux"
backend = get_backend(config)

# Cleanup (important for multi-model systems)
unload_current_backend()  # Explicitly frees VRAM
```

### VRAM Cleanup

Both SDXL and FLUX backends implement `unload_pipeline()`:

1. **Delete pipeline object** (Python refcounting)
2. **Clear CUDA cache** (`torch.cuda.empty_cache()`)
3. **Run garbage collection** (`gc.collect()`)

**Explicit cleanup is critical** in long-running processes or systems that cycle through multiple models (e.g., parse → generate SDXL → generate FLUX → evaluate).

## ControlNet Support

### Current Implementation

- **Supported:** SDXL backend with depth ControlNet
- **Not supported:** FLUX (no compatible ControlNet models)
- **Graceful fallback:** If ControlNet requested but unsupported, error is raised with clear message

### How It Works

1. **Layout parsing** (`constraint_builder.py`):
   - Story layout specifies character positions (left/center/right) and depths (foreground/midground/background)
   - Optional Groq-based layout planning generates layout JSON

2. **Control image generation** (`controlled_generator.py`):
   - `layout_to_control_image()` converts layout to grayscale depth map
   - Character positions encoded as x-coordinates; depths as brightness values
   - Output is PIL Image compatible with ControlNet

3. **ControlNet generation** (SDXL backend):
   - Backend loads separate ControlNet pipeline if not cached
   - Passes control image + prompt to SDXL ControlNet pipeline
   - Conditioning scale controls strength (0.0 = ignore, 1.0 = strict)

### Limitations

- **Depth only:** No canny, pose, or segmentation ControlNets
- **SDXL only:** FLUX ControlNet not yet available
- **Single model:** Cannot combine multiple ControlNets
- **Layout dependency:** Requires valid layout; skipped gracefully if layout generation fails

## Development Guide

### Adding a New Backend

To add support for a new model (e.g., SD3, PixArt):

1. **Create backend class** (`generator/backends/new_backend.py`):
   ```python
   from generator.backends.base import BaseGeneratorBackend
   from PIL import Image

   class NewModelBackend(BaseGeneratorBackend):
       backend_name = "new_model"

       def load_pipeline(self):
           # Load model and return pipeline
           pass

       def unload_pipeline(self):
           # Delete pipeline, clear CUDA cache, run gc.collect()
           pass

       def generate(self, prompt: str, seed: int, config: dict) -> Image.Image:
           # Generate single image from prompt + seed
           # Extract parameters from config (steps, guidance, etc.)
           pass

       def supports_controlnet(self) -> bool:
           # Return True if this backend supports ControlNet
           return False

       def update_config(self, generation_config: dict, full_config: dict) -> None:
           # Update internal config without reloading pipeline
           self.generation_config = generation_config
           self.full_config = full_config
   ```

2. **Register in factory** (`generator/backends/backend_factory.py`):
   ```python
   from .new_backend import NewModelBackend

   _BACKEND_REGISTRY: dict[str, type[BaseGeneratorBackend]] = {
       "sdxl": SDXLBackend,
       "flux": FluxBackend,
       "new_model": NewModelBackend,  # Add here
   }
   ```

3. **Add config section** (`config.yaml`):
   ```yaml
   backends:
     new_model:
       model_id: "namespace/new-model-name"
       # ... model-specific settings
   ```

4. **Test via factory:**
   ```python
   config = {"generation": {"backend": "new_model"}, "backends": {...}}
   backend = get_backend(config)
   images = backend.generate("Test prompt", seed=42, config={...})
   ```

### Design Principles

- **Keep orchestration layer clean:** `image_generator.py` should remain backend-agnostic
- **Avoid duplicated logic:** All generation logic lives in backend classes
- **Explicit lifecycle:** Implement `load_pipeline()` and `unload_pipeline()` correctly
- **Consistent metadata:** Use `GeneratedImage` dataclass with backend_name, precision, quantized, generation_time
- **Feature flags:** Use `supports_controlnet()` to gate unsupported functionality gracefully
- **Lazy loading:** Load pipelines only on first `generate()` call, not on backend instantiation

## Logging & Debugging

### Generation Logs

The generator logs key information at INFO level:

```
generate_images | backend=sdxl model=stabilityai/stable-diffusion-xl-base-1.0 precision=fp16 quantized=False
generate_images | n=3 steps=30 guidance=7.5 768x768
Reusing cached SDXL pipeline for stabilityai/stable-diffusion-xl-base-1.0
Selected backend=sdxl
[GENERATOR] Prompt: A cinematic scene...
```

### Backend Logs

Backend logs include:
- Pipeline loading events (`Loading SDXL pipeline`, `Loading FLUX pipeline`)
- Device selection (`SDXL device=cuda dtype=float16`)
- VRAM snapshots (`CUDA memory: allocated=8192 MiB reserved=10240 MiB`)
- Optimization info (`xformers memory-efficient attention enabled`)
- Errors and warnings (`No GPU detected — SDXL will run on CPU. Generation will be very slow.`)

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all DEBUG-level logs from generator.* will print
from generator import generate_images
```

### ControlNet Debugging

If `save_debug_control: true` in `config.yaml`, control images are saved to the output directory for inspection.

## Limitations

### Hardware

- **GPU requirement:** Practical use requires a modern GPU with 14+ GB VRAM
  - Laptop GPUs often insufficient
  - CPUs supported but unreasonably slow
  - No multi-GPU distribution (single device only)

### FLUX Quantization

- **fp8 requires CUDA + bitsandbytes:** Not compatible with CPU or Apple Silicon
- **Quantization overhead:** First load quantizes model (~30 seconds); subsequent loads faster
- **Quality:** Minor quality degradation reported in some cases; acceptable for most use cases

### ControlNet

- **SDXL only:** FLUX ControlNet models not yet available (may change)
- **Depth only:** No alternative ControlNet types implemented
- **Layout-dependent:** Fails gracefully if layout is invalid, but skips ControlNet entirely

### Configuration vs Code

- Some `config.yaml` settings describe intent but are overridden by code defaults
  - Example: `evaluator.clip_model` is ignored; OpenCLIP defaults used instead
- No automatic config validation; invalid settings fail at runtime
- Backend switching requires explicit config change (no CLI flag)

### API Costs

- LLM parsing and layout planning are **paid services** (Gemini, Groq, NVIDIA)
- Repeated calls add up quickly
- Consider caching (prompt_cache in config.yaml)

### Dependencies

- **Large dependency set:** `requirements.txt` includes packages not directly used by main code
- **Environment-specific:** CUDA/driver compatibility can be fragile
- **Installation friction:** May require manual fixes on some systems

## Future Work

Potential improvements (not currently implemented):

- **SD3 support:** New Stability AI diffusion model backend
- **PixArt backend:** Fast diffusion model for rapid iteration
- **Improved ControlNet integration:** Support for additional ControlNet types (canny, pose, etc.)
- **FLUX ControlNet:** When compatible models become available
- **Distributed inference:** Multi-GPU support for large batches
- **Memory optimization:** Attention mechanisms beyond xformers (flash-attn, etc.)
- **Model merging:** Support for LoRA-merged base models
- **Streaming output:** Generate images in real-time rather than batch mode

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** `CUDA out of memory` or `torch.cuda.OutOfMemoryError`

**Solutions:**
1. Reduce `generation.height` and `generation.width` (e.g., 768 instead of 1024)
2. Reduce `generation.num_candidates` (fewer images per scene)
3. Switch to FLUX fp8 quantization (requires CUDA + bitsandbytes)
4. Disable ControlNet if enabled
5. Ensure previous processes have cleaned up; call `unload_current_backend()`

### Slow Generation

**Symptoms:** Generation takes 5+ minutes per image

**Likely causes:**
- Running on CPU (expected to be slow)
- GPU not being used (check `nvidia-smi`)
- Very high `generation.steps` value

**Solutions:**
1. Verify GPU is available: `torch.cuda.is_available()`
2. Ensure CUDA drivers are up to date
3. Reduce `generation.steps` to 20–25
4. Check VRAM usage; other processes may be consuming memory

### API Key Errors

**Symptoms:** `API_KEY not found` or authentication failures

**Solutions:**
1. Verify environment variables are set:
   ```bash
   echo $GEMINI_API_KEY  # Should not be empty
   ```
2. Check that keys are valid in their respective consoles
3. Ensure keys have appropriate scopes/permissions

### Backend Mismatch

**Symptoms:** `ValueError: Unknown generation backend: 'sdxl3'`

**Solution:**
- Check `config.yaml`: `generation.backend` must be "sdxl" or "flux" (case-sensitive, lowercase)
- Verify backend section exists under `backends.`

## License

No `LICENSE` file is present in this repository. All rights and usage terms are unspecified until the authors add a license.
