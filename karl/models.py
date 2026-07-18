"""The model catalog KARL can serve on a GPU box.

One row per model: its weights, runtime (vLLM FP8 or llama.cpp GGUF), VRAM
floor, and how context scales with VRAM. ``karl gpu model <key>`` picks a row;
``karl gpu ssh …`` serves whatever is selected. GLM is the default — the
operator's daily driver.

NOTE: rows are ASPIRATIONAL — ``repo``/``served_name`` are the intended Hugging
Face repos and served names, and a serve fails at launch if a repo doesn't
resolve on the box. Treat the catalog as a starting set to confirm against real
repos, and edit this file to add your own.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    repo: str
    served_name: str      # the `model` string the OpenAI client must send
    min_total_gb: int     # VRAM floor: weights + runtime overhead
    max_model_len: int    # longest context the model supports
    context_tiers: list   # [(total_gb_threshold, max_model_len), ...] first fit wins
    context_beyond: int   # context when VRAM exceeds every threshold
    weights_note: str     # shown while the box downloads
    server: str = "vllm"          # "vllm" (FP8 safetensors) or "llama_cpp" (GGUF)
    quantization: str = "fp8"
    tool_call_parser: str = "hermes"
    gguf_file: str | None = None  # exact GGUF filename in the repo
    gguf_quant: str | None = None  # or a quant tag llama.cpp resolves (repo:QUANT)
    tokenizer: str | None = None

    @property
    def is_gguf(self) -> bool:
        return self.gguf_file is not None or self.gguf_quant is not None


GLM = ModelSpec(
    key="glm",
    label="GLM-4.7-Flash (HauhauCS Balanced, uncensored) · FP16 GGUF",
    repo="HauhauCS/GLM-4.7-Flash-Uncensored-HauhauCS-Balanced",
    served_name="glm-4.7-flash",
    min_total_gb=66,
    max_model_len=131072,
    context_tiers=[(80, 16384), (88, 32768), (100, 65536), (120, 98304)],
    context_beyond=131072,
    weights_note="first run downloads the ~62GB FP16 GGUF",
    server="llama_cpp",
    quantization="gguf",
    gguf_file="GLM-4.7-Flash-Uncensored-HauhauCS-Balanced-FP16.gguf",
)

HERMES = ModelSpec(
    key="hermes",
    label="Hermes-4.3-36B (NousResearch) · FP8",
    repo="NousResearch/Hermes-4.3-36B",
    served_name="NousResearch/Hermes-4.3-36B",
    min_total_gb=44,
    max_model_len=524288,
    context_tiers=[(56, 16384), (72, 32768), (96, 65536), (120, 131072), (168, 196608)],
    context_beyond=262144,
    weights_note="first run downloads ~37GB of FP8 weights",
    server="vllm",
    quantization="fp8",
    tool_call_parser="hermes",
)

QWEN_OFFICIAL = ModelSpec(
    key="qwen-official",
    label="Qwen3.6-27B (Alibaba, official) · FP8",
    repo="Qwen/Qwen3.6-27B",
    served_name="qwen3.6-27b-official",
    min_total_gb=30,
    max_model_len=262144,
    context_tiers=[(40, 32768), (56, 65536), (80, 131072), (140, 196608)],
    context_beyond=262144,
    weights_note="first run downloads ~27GB and quantizes to FP8 on the fly",
    server="vllm",
    quantization="fp8",
    tool_call_parser="hermes",
)

QWEN = ModelSpec(
    key="qwen",
    label="Qwen3.6-27B (HauhauCS Balanced, uncensored) · Q5_K_P GGUF",
    repo="HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced",
    served_name="qwen3.6-27b",
    min_total_gb=22,
    max_model_len=131072,
    context_tiers=[(28, 16384), (40, 32768), (56, 65536), (96, 98304)],
    context_beyond=131072,
    weights_note="first run downloads the ~19GB Q5_K_P GGUF",
    server="llama_cpp",
    quantization="gguf",
    gguf_file="Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q5_K_P.gguf",
)

QWEN_40B = ModelSpec(
    key="qwen-40b",
    label="Qwen3.6-40B (DavidAU, Opus-Deckard Heretic, uncensored) · Q5_K_M GGUF",
    repo="DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF",
    served_name="qwen3.6-40b",
    min_total_gb=30,
    max_model_len=131072,
    context_tiers=[(36, 16384), (48, 32768), (72, 65536), (120, 98304)],
    context_beyond=131072,
    weights_note="first run downloads the ~28GB Q5_K_M GGUF",
    server="llama_cpp",
    quantization="gguf",
    gguf_quant="Q5_K_M",
)

CATALOG: dict[str, ModelSpec] = {
    GLM.key: GLM,
    HERMES.key: HERMES,
    QWEN_OFFICIAL.key: QWEN_OFFICIAL,
    QWEN.key: QWEN,
    QWEN_40B.key: QWEN_40B,
}
DEFAULT_KEY = GLM.key


def model_list() -> list:
    return list(CATALOG.values())


def get_spec(key: str) -> ModelSpec:
    return CATALOG.get(key or DEFAULT_KEY, GLM)
