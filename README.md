# story2img

## Overview

This repository implements a **story-to-image pipeline**: raw story text is structured by an **LLM** (characters, scenes, style), then **Stable Diffusion XL (SDXL)** generates **multiple candidate images per scene**, and an **OpenCLIP-based evaluator** scores candidates and selects one per scene. An optional path combines SDXL with **ControlNet** using a **layout** derived from constraints (including optional **Groq** layout planning). A **memory** module stores per-character reference images across scenes in the “v2” pipeline.

**Within a few seconds:** it is research/engineering code for automated illustration from text—not a polished end-user app.

---

## ⚠️ Resource requirements

- **GPU expectations:** Image generation uses **SDXL** (and optionally **SDXL + ControlNet**). The default `config.yaml` uses **1024×1024** outputs. In practice this class of stack typically needs a **high-end GPU with a large amount of VRAM** (often **around 16 GB or more** for comfortable use at that resolution; exact headroom depends on batching, ControlNet, and system overhead). **Many laptops and consumer machines will not run this reliably.**
- **CPU / Apple Silicon:** The generator can fall back to **CPU** or **MPS**, but the code explicitly warns that **CPU SDXL is extremely slow** and not practical for normal use.
- **Remote LLM APIs:** Parsing (and optional layout planning) require **network access** and **API keys** for the configured providers (see Setup).

**Local execution may not be feasible** for you. Treat **reading the code and docstrings**, **running LLM-only checks**, or **using a remote GPU workstation** as realistic alternatives—not assuming everything runs on a typical laptop.

---

## Features

*(From modules and docstrings in this repo—no extra products are claimed.)*

- **LLM story parser** (`llm/parser.py`): turns a story string into validated JSON (`characters`, `scenes`, `style`) with retries.
- **Provider clients** (`llm/`): **Gemini**, **Groq**, **NVIDIA** (OpenAI-compatible) behind a shared `LLMBase` / `LLMResponse` interface; factory in `llm/__init__.py` (`build_llm_client`).
- **Phase 5 pipeline** (`pipeline/story_pipeline.py`): per scene—generate N candidates with SDXL, score with CLIP, pick best; **no memory** (`reference_image` always `None` for evaluation).
- **Phase 7 pipeline** (`pipeline/story_pipeline_v2.py`): **memory** for characters, **constraint-based prompts**, optional **prompt cache**, optional **ControlNet** generation, evaluation still uses **scene description** (not the built prompt text).
- **Constraint builder** (`constraints/constraint_builder.py`): structured constraints, optional **Groq** layout with **disk cache** and **fallback layout**, prompt compression / assembly.
- **Image generation** (`generator/image_generator.py`, `generator/controlled_generator.py`): SDXL; optional ControlNet with a **layout-derived control image**.
- **Evaluator** (`evaluator/clip_evaluator.py`): **OpenCLIP** singleton; text–image and image–image similarity; weighted **scene / identity / temporal** scoring; optional on-disk **text embedding cache**.
- **Memory** (`memory/memory_manager.py`): in-process store of **reference images** and **first-write descriptions** per character.
- **Sanity / dev scripts** (`utils/sanity_check_*.py`): small `__main__` harnesses for parser, generator, evaluator, constraints, pipelines, memory—not a single official CLI.

---

## Tech stack

- **Language:** Python 3 (implicit from typing syntax and dependencies).
- **Core libraries used by the Python modules:** **PyTorch**, **diffusers** (SDXL / ControlNet), **Pillow**, **open_clip_torch**, **PyYAML**, **python-dotenv**, provider SDKs (**google-generativeai**, **groq**, **openai**), **httpx** (transitive), **tqdm**/Hugging Face stack via diffusers/transformers.
- `**requirements.txt`:** pins a **large** dependency set (including **CUDA-related** NVIDIA wheels and other packages). **Not every top-level package in that file is imported by the `.py` files in this repository** (e.g. no `gradio` / `fastapi` usage appears in the tracked source). Treat `requirements.txt` as an environment lockfile that may need trimming or fixes on your machine.

---

## Setup / installation

1. **Clone** the repository and open a terminal at the repo root.
2. **Python environment:** Use a **virtual environment** (recommended). Version is not declared in-repo; use a current Python compatible with the pinned stack (see `requirements.txt`).
3. **Install dependencies:**
  ```bash
   pip install -r requirements.txt
  ```
   **Caveat:** One line in `requirements.txt` may reference a **local conda build path** for `packaging`. If installation fails, replace that entry with a normal **PyPI** `packaging` pin.
4. **Configuration:** Copy or edit `**config.yaml`** at the repo root (generation size, LLM provider, evaluator weights, ControlNet, layout, prompt cache, etc.).
5. **API keys (remote LLM):** As implemented in `llm/*_client.py`, set environment variables as appropriate, e.g.:
  - `GEMINI_API_KEY` (Gemini)
  - `GROQ_API_KEY` (Groq)
  - `NVIDIA_API_KEY` (NVIDIA OpenAI-compatible endpoint)
6. **GPU:** Install a **PyTorch build that matches your CUDA/driver** (or use MPS on Apple Silicon). **Without a suitable GPU, full pipeline runs are usually not realistic.**

If you **cannot** satisfy GPU and dependency constraints, **do not assume a successful full install**; you can still use the **Code understanding** path below.

---

## Usage

### A. Code understanding (primary path)

- **Docstrings:** Functions and public methods are documented in **NumPy style** (summary, parameters, returns, notes, edge cases). Start with:
  - `pipeline/story_pipeline.py` — `run_pipeline`
  - `pipeline/story_pipeline_v2.py` — `run_pipeline_v2`, `_analyze_prompt`
  - `llm/parser.py` — `StoryParser`, helpers
  - `generator/image_generator.py` — `generate_images`, `_load_pipeline`
  - `generator/controlled_generator.py` — ControlNet path
  - `evaluator/clip_evaluator.py` — `score_candidates`, embedding cache
  - `constraints/constraint_builder.py` — layout and prompt building
  - `memory/memory_manager.py` — storage semantics
- **Entry points (examples, not a single product CLI):** run modules from the repo root with `python -m` or `python path/to/script.py`, e.g.:
  - `utils/sanity_check_parser.py` — parser over `data/test_stories` (LLM calls; see script docstring)
  - `utils/sanity_check_constraint.py` — constraints from cached parser JSON
  - `utils/sanity_check_generator.py` — SDXL smoke test (**heavy**)
  - `utils/sanity_check_pipeline_full.py` / `utils/sanity_check_pipeline_v2.py` — end-to-end batch runs (**heavy**)

Use these to trace **data flow**: text → structured scenes → prompts → images → CLIP scores → selected images (+ memory updates in v2).

### B. Execution (if applicable)

**Only attempt if you have adequate GPU resources and API keys.**

1. Configure `config.yaml` and environment variables.
2. From repo root (with `PYTHONPATH` including the project, or run scripts that modify `sys.path` as they do today), run the relevant `utils/sanity_check_*.py` or import `run_pipeline` / `run_pipeline_v2` in your own driver script.
3. **Warnings:** Default resolution and candidate counts are **VRAM-intensive**. ControlNet loads **additional** weights. Expect long runtimes and possible OOM on smaller GPUs.

There is **no** documented one-command “production” launcher in this repo beyond these scripts.

---

## Project structure

```text
story2img/
├── config.yaml              # Master YAML config (LLM, generation, evaluator, layout, etc.)
├── requirements.txt         # Pinned dependencies (large; see Tech stack)
├── data/
│   └── test_stories/        # Sample .txt stories for sanity scripts
├── llm/                     # LLM base, providers, story parser, client factory
├── pipeline/                # run_pipeline (no memory), run_pipeline_v2 (memory + constraints)
├── generator/               # SDXL + optional ControlNet generation
├── evaluator/               # OpenCLIP scoring
├── constraints/             # Constraint and prompt construction (+ optional layout LLM)
├── memory/                  # In-memory per-character reference store
├── utils/                   # sanity_check_* scripts (__main__)
├── cache/                   # Used at runtime (e.g. layout / embedding caches; may be created)
└── logs/                    # Written by scripts and constraint builder file logging
```

---

## Design & architecture notes

*(Strictly from code behaviour.)*

- **Two pipeline stages:** `run_pipeline` (Phase 5) never passes a reference image into the evaluator; `run_pipeline_v2` pulls reference images from `MemoryManager`, evaluates with **scene `description*`*, not the final generator prompt string.
- **Candidate generation:** Each scene requests **N images** (`num_candidates` in config); the evaluator returns `**best_index`** and `**best_image**`.
- **Seeds:** Generator resolves seeds via `**incremental`** or `**random**` strategy from config.
- **Constraint path:** Structured constraints → prompt string; optional **Groq** layout JSON with validation and **fallback_layout**; **layout cache** on disk under `cache/`.
- **ControlNet:** Optional; requires valid `layout` in constraints when enabled; merges `controlnet` block into generation config for `generate_images_controlled`.
- **CLIP cache:** Text embeddings may be cached in `cache/embedding_cache.json` (keyed by hash of text).

---

## Limitations

- **Hardware:** Needs a **strong GPU** for practical SDXL (+ optional ControlNet) use; **CPU is supported but not viable** for serious runs.
- **No packaged app:** No single documented CLI product; usability is **library + ad hoc scripts**.
- **No in-repo license file** (see below).
- **No Docker / cloud deployment** definitions in this repository.
- **API costs and quotas:** LLM (and optional layout) calls are **paid/rate-limited** per provider.
- **Config vs code:** Some `config.yaml` fields describe intent (e.g. evaluator model name); the **evaluator implementation** is driven by **OpenCLIP** defaults in `evaluator/clip_evaluator.py` unless you change the code.
- `**requirements.txt`:** May be **over-broad** or **environment-specific**; expect possible install friction.

---

## License

No `LICENSE` file or license header was found in this repository. **All rights and usage terms are unspecified** until the authors add a license.
