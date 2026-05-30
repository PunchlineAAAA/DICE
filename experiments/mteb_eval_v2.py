import argparse
import inspect
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_RECOMMENDED_MTEB_VERSION = (2, 2, 0)
BRIGHT_SUBSET_MAPPING = {
    "BrightBiology": "biology",
    "BrightEconomics": "economics",
    "BrightStackOverflow": "stackoverflow",
    "BrightLeetcode": "leetcode",
    "BrightPony": "pony",
    "BrightAops": "aops",
    "BrightTheoremqaTheorems": "theoremqa_theorems",
    "BrightTheoremqaQuestions": "theoremqa_questions",
}
TASK_PRESETS = {
    "followir": [
        "News21InstructionRetrieval",
        "Core17InstructionRetrieval",
        "Robust04InstructionRetrieval",
    ],
    "longembed": [
        "LEMBNarrativeQARetrieval",
        "LEMBNeedleRetrieval",
        "LEMBPasskeyRetrieval",
        "LEMBQMSumRetrieval",
        "LEMBSummScreenFDRetrieval",
        "LEMBWikimQARetrieval",
    ],
    "bright_core": [
        "BrightAops",
        "BrightBiology",
        "BrightEconomics",
        "BrightLeetcode",
        "BrightPony",
        "BrightStackOverflow",
        "BrightTheoremqaTheorems",
        "BrightTheoremqaQuestions",
    ],
    "bright_full": [
        "BrightAops",
        "BrightBiology",
        "BrightEconomics",
        "BrightLeetcode",
        "BrightPony",
        "BrightStackOverflow",
        "BrightTheoremqaTheorems",
        "BrightTheoremqaQuestions",
        "BrightEarthScience",
        "BrightPsychology",
        "BrightRobotics",
        "BrightSustainableLiving",
    ],
}
DOC_ENCODING_PRESETS = {
    "single": {
        "doc_chunk_size": 0,
        "doc_chunk_overlap": 0,
        "doc_agg_mode": "mean",
        "doc_top_k": 3,
        "doc_softmax_temperature": 1.0,
        "disable_chunking_for_short_docs": False,
    },
    "chunk128_mean": {
        "doc_chunk_size": 128,
        "doc_chunk_overlap": 32,
        "doc_agg_mode": "mean",
        "doc_top_k": 3,
        "doc_softmax_temperature": 1.0,
        "disable_chunking_for_short_docs": False,
    },
    "chunk256_mean": {
        "doc_chunk_size": 256,
        "doc_chunk_overlap": 64,
        "doc_agg_mode": "mean",
        "doc_top_k": 3,
        "doc_softmax_temperature": 1.0,
        "disable_chunking_for_short_docs": False,
    },
    "chunk256_mean_skipshort": {
        "doc_chunk_size": 256,
        "doc_chunk_overlap": 64,
        "doc_agg_mode": "mean",
        "doc_top_k": 3,
        "doc_softmax_temperature": 1.0,
        "disable_chunking_for_short_docs": True,
    },
    "chunk256_nooverlap": {
        "doc_chunk_size": 256,
        "doc_chunk_overlap": 0,
        "doc_agg_mode": "mean",
        "doc_top_k": 3,
        "doc_softmax_temperature": 1.0,
        "disable_chunking_for_short_docs": False,
    },
    "chunk512_mean": {
        "doc_chunk_size": 512,
        "doc_chunk_overlap": 128,
        "doc_agg_mode": "mean",
        "doc_top_k": 3,
        "doc_softmax_temperature": 1.0,
        "disable_chunking_for_short_docs": False,
    },
    "chunk512_mean_skipshort": {
        "doc_chunk_size": 512,
        "doc_chunk_overlap": 128,
        "doc_agg_mode": "mean",
        "doc_top_k": 3,
        "doc_softmax_temperature": 1.0,
        "disable_chunking_for_short_docs": True,
    },
    "chunk512_nooverlap": {
        "doc_chunk_size": 512,
        "doc_chunk_overlap": 0,
        "doc_agg_mode": "mean",
        "doc_top_k": 3,
        "doc_softmax_temperature": 1.0,
        "disable_chunking_for_short_docs": False,
    },
    "chunk512_lmk_nooverlap": {
        "doc_chunk_size": 512,
        "doc_chunk_overlap": 0,
        "doc_agg_mode": "lmk_mean",
        "doc_top_k": 3,
        "doc_softmax_temperature": 1.0,
        "disable_chunking_for_short_docs": False,
    },
    "chunk1024_lmk_nooverlap": {
        "doc_chunk_size": 1024,
        "doc_chunk_overlap": 0,
        "doc_agg_mode": "lmk_mean",
        "doc_top_k": 3,
        "doc_softmax_temperature": 1.0,
        "disable_chunking_for_short_docs": False,
    },
    "chunk1024_nooverlap_token_reset_mean": {
        "doc_chunk_size": 1024,
        "doc_chunk_overlap": 0,
        "doc_agg_mode": "mean",
        "doc_top_k": 3,
        "doc_softmax_temperature": 1.0,
        "disable_chunking_for_short_docs": False,
        "dice_position_mode": "reset",
        "dice_chunk_input_mode": "token_ids",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM2Vec checkpoint with a stable official MTEB v2 retrieval pipeline."
    )
    parser.add_argument("--base_model_name_or_path", required=True)
    parser.add_argument("--peft_model_name_or_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--benchmark_name", default="MTEB(eng, v2)")
    parser.add_argument(
        "--preset",
        choices=sorted(TASK_PRESETS.keys()),
        default=None,
        help="Named task bundle. If set, overrides --task_names and --benchmark_name.",
    )
    parser.add_argument(
        "--task_names",
        default=None,
        help="Optional comma-separated task names. If set, overrides --benchmark_name.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--similarity_mode", choices=["dot", "cosine"], default="cosine")
    parser.add_argument(
        "--save_summary_json",
        action="store_true",
        help="Write a compact task/result summary next to MTEB outputs.",
    )
    parser.add_argument(
        "--doc_encoding_mode",
        choices=sorted(DOC_ENCODING_PRESETS.keys()),
        default="single",
        help="Stable document encoding mode for evaluation.",
    )
    parser.add_argument(
        "--doc_chunk_size",
        type=int,
        default=None,
        help="Optional override for document chunk size. Use 0 to disable chunking.",
    )
    parser.add_argument(
        "--doc_chunk_overlap",
        type=int,
        default=None,
        help="Optional override for token overlap between adjacent document chunks.",
    )
    parser.add_argument(
        "--doc_agg_mode",
        choices=["mean", "max", "norm_topk_mean", "norm_softmax_mean", "lmk_mean"],
        default=None,
        help="Optional override for document chunk aggregation mode.",
    )
    parser.add_argument(
        "--doc_top_k",
        type=int,
        default=None,
        help="Optional override for top-k used by norm_topk_mean aggregation.",
    )
    parser.add_argument(
        "--doc_softmax_temperature",
        type=float,
        default=None,
        help="Optional override for temperature used by norm_softmax_mean aggregation.",
    )
    parser.add_argument(
        "--disable_chunking_for_short_docs",
        action="store_true",
        help="If set, short documents stay on the single-vector path even when chunking is enabled.",
    )
    parser.add_argument(
        "--dice_position_mode",
        choices=["reset", "absolute_offset"],
        default=None,
        help="DICE chunk position-id ablation.",
    )
    parser.add_argument(
        "--dice_chunk_input_mode",
        choices=["text", "token_ids"],
        default=None,
        help="Whether DICE chunks are decoded text chunks or direct token-id slices.",
    )
    parser.add_argument(
        "--longembed_local_dir",
        default=None,
        help="Optional local path for a manually downloaded dwzhu/LongEmbed dataset clone.",
    )
    return parser.parse_args()


def parse_version_tuple(version_text: str) -> tuple[int, ...]:
    parts = []
    for chunk in version_text.replace("-", ".").split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def version_gte(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> bool:
    max_len = max(len(lhs), len(rhs))
    lhs = lhs + (0,) * (max_len - len(lhs))
    rhs = rhs + (0,) * (max_len - len(rhs))
    return lhs >= rhs


def load_official_mteb():
    original_sys_path = sys.path.copy()
    project_root_resolved = PROJECT_ROOT.resolve()
    sys.path = [
        path
        for path in sys.path
        if Path(path or ".").resolve() != project_root_resolved
    ]
    try:
        import mteb
    except ModuleNotFoundError as exc:
        if exc.name != "mteb":
            raise RuntimeError(
                "Official MTEB is installed but failed to import a dependency. "
                f"Original error: {exc!r}"
            ) from exc
        raise ImportError(
            "Official MTEB is not installed. Install `mteb>=2.2.0` in the active environment."
        ) from exc
    finally:
        sys.path = original_sys_path

    try:
        installed_version = version("mteb")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Could not read installed MTEB package metadata. Please reinstall `mteb>=2.2.0`."
        ) from exc

    parsed_version = parse_version_tuple(installed_version)
    if parsed_version[0] < 2:
        raise RuntimeError(
            f"Installed MTEB version is {installed_version}, but this script requires v2 or newer."
        )
    if not version_gte(parsed_version, MIN_RECOMMENDED_MTEB_VERSION):
        raise RuntimeError(
            "Installed MTEB version is "
            f"{installed_version}, but this script requires mteb>="
            f"{'.'.join(map(str, MIN_RECOMMENDED_MTEB_VERSION))}."
        )
    return mteb, installed_version


def patch_mteb_cached_lemb_configs() -> bool:
    """Force cached LEMB retrieval datasets to use corpus/qrels/queries configs.

    Some MTEB versions may initialize cached LEMB retrieval tasks with the
    `default` config even when the local cache is organized under the standard
    retrieval configs. This patch only affects cached `mteb/LEMB*` tasks.
    """
    try:
        from mteb.abstasks import retrieval_dataset_loaders
    except Exception:
        return False

    loader_cls = getattr(retrieval_dataset_loaders, "RetrievalDatasetLoader", None)
    if loader_cls is None or getattr(loader_cls, "_diffembed_lemb_cache_patch", False):
        return False

    original_init = loader_cls.__init__

    def patched_init(
        self,
        hf_repo: str,
        revision: str,
        trust_remote_code: bool = False,
        split: str = "test",
        config: str | None = None,
    ):
        original_init(
            self,
            hf_repo=hf_repo,
            revision=revision,
            trust_remote_code=trust_remote_code,
            split=split,
            config=config,
        )
        if (
            isinstance(hf_repo, str)
            and hf_repo.startswith("mteb/LEMB")
            and self.dataset_configs == ["default"]
        ):
            self.dataset_configs = ["corpus", "qrels", "queries"]

    loader_cls.__init__ = patched_init
    loader_cls._diffembed_lemb_cache_patch = True
    return True


def load_llm2vec():
    try:
        from llm2vec import LLM2Vec
    except ImportError as exc:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        try:
            from llm2vec import LLM2Vec
        except ImportError:
            raise ImportError(
                "Could not import `llm2vec`. Run `pip install -e .` from the DiffEmbed root first."
            ) from exc
    return LLM2Vec


def load_model_meta_class(mteb_module):
    candidate_paths = [
        ("models.model_meta", "ModelMeta"),
        ("model_meta", "ModelMeta"),
    ]
    for module_name, attribute_name in candidate_paths:
        try:
            module = __import__(f"mteb.{module_name}", fromlist=[attribute_name])
            return getattr(module, attribute_name)
        except Exception:
            continue
    if hasattr(mteb_module, "ModelMeta"):
        return mteb_module.ModelMeta
    return None


def load_result_cache_class(mteb_module):
    if hasattr(mteb_module, "ResultCache"):
        return mteb_module.ResultCache
    try:
        from mteb.cache import ResultCache

        return ResultCache
    except Exception:
        return None


def resolve_tasks(mteb_module, preset: str | None, task_names: str | None, benchmark_name: str):
    if preset:
        task_list = TASK_PRESETS[preset]
        if hasattr(mteb_module, "get_tasks"):
            return mteb_module.get_tasks(tasks=task_list)
        if hasattr(mteb_module, "get_task"):
            return mteb_module.get_task(task_list)
        raise AttributeError("Installed MTEB does not expose `get_tasks` or `get_task`.")

    if task_names:
        task_list = [task.strip() for task in task_names.split(",") if task.strip()]
        if hasattr(mteb_module, "get_tasks"):
            return mteb_module.get_tasks(tasks=task_list)
        if hasattr(mteb_module, "get_task"):
            return mteb_module.get_task(task_list)
        raise AttributeError("Installed MTEB does not expose `get_tasks` or `get_task`.")

    if hasattr(mteb_module, "get_benchmark"):
        return mteb_module.get_benchmark(benchmark_name)
    raise AttributeError("Installed MTEB does not expose `get_benchmark`.")


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, prediction_dir


def _to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "model_dump"):
        try:
            return _to_jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _to_jsonable(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)


def iter_task_names(selected_tasks) -> List[str]:
    tasks = selected_tasks.tasks if hasattr(selected_tasks, "tasks") else selected_tasks
    names = []
    for task in tasks:
        if getattr(task, "metadata", None) is not None:
            names.append(task.metadata.name)
        else:
            names.append(str(task))
    return names


def write_summary_json(output_dir: Path, results, selected_tasks, eval_config: dict[str, Any]) -> Path:
    summary_path = output_dir / "summary.json"
    payload = {
        "tasks": iter_task_names(selected_tasks),
        "eval_config": _to_jsonable(eval_config),
        "results": _to_jsonable(results),
    }
    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)
    return summary_path


def default_instruction_resolver(task_name: str) -> str:
    try:
        from mteb.models.instructions import task_to_instruction  # type: ignore
    except Exception:
        try:
            from mteb.mteb.models.instructions import task_to_instruction  # type: ignore
        except Exception:
            return ""
    try:
        return task_to_instruction(task_name)
    except Exception:
        return ""


def load_task_to_instructions(path: str | None):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def resolve_doc_encoding_config(mode: str) -> dict[str, Any]:
    if mode not in DOC_ENCODING_PRESETS:
        raise ValueError(f"Unknown doc encoding mode: {mode}")
    config = dict(DOC_ENCODING_PRESETS[mode])
    config.setdefault("dice_position_mode", "reset")
    config.setdefault("dice_chunk_input_mode", "text")
    return config


def apply_doc_encoding_overrides(base_config: dict[str, Any], args) -> dict[str, Any]:
    config = dict(base_config)
    if args.doc_chunk_size is not None:
        config["doc_chunk_size"] = args.doc_chunk_size
    if args.doc_chunk_overlap is not None:
        config["doc_chunk_overlap"] = args.doc_chunk_overlap
    if args.doc_agg_mode is not None:
        config["doc_agg_mode"] = args.doc_agg_mode
    if args.doc_top_k is not None:
        config["doc_top_k"] = args.doc_top_k
    if args.doc_softmax_temperature is not None:
        config["doc_softmax_temperature"] = args.doc_softmax_temperature
    if args.disable_chunking_for_short_docs:
        config["disable_chunking_for_short_docs"] = True
    if args.dice_position_mode is not None:
        config["dice_position_mode"] = args.dice_position_mode
    if args.dice_chunk_input_mode is not None:
        config["dice_chunk_input_mode"] = args.dice_chunk_input_mode

    if config["doc_chunk_size"] < 0:
        raise ValueError("doc_chunk_size must be >= 0")
    if config["doc_chunk_overlap"] < 0:
        raise ValueError("doc_chunk_overlap must be >= 0")
    if config["doc_chunk_size"] > 0 and config["doc_chunk_overlap"] >= config["doc_chunk_size"]:
        raise ValueError("doc_chunk_overlap must be smaller than doc_chunk_size")
    if config["doc_top_k"] <= 0:
        raise ValueError("doc_top_k must be > 0")
    if config["doc_softmax_temperature"] <= 0:
        raise ValueError("doc_softmax_temperature must be > 0")
    if config["dice_position_mode"] not in {"reset", "absolute_offset"}:
        raise ValueError("Unknown dice_position_mode")
    if config["dice_chunk_input_mode"] not in {"text", "token_ids"}:
        raise ValueError("Unknown dice_chunk_input_mode")
    if config["dice_position_mode"] == "absolute_offset" and config["dice_chunk_input_mode"] != "token_ids":
        raise ValueError("absolute_offset currently requires dice_chunk_input_mode=token_ids")

    return config


def override_local_dataset_paths(selected_tasks, longembed_local_dir: str | None) -> list[str]:
    if not longembed_local_dir:
        return []

    local_dir = Path(longembed_local_dir).expanduser().resolve()
    if not local_dir.exists():
        raise FileNotFoundError(f"LongEmbed local dir does not exist: {local_dir}")

    tasks = selected_tasks.tasks if hasattr(selected_tasks, "tasks") else selected_tasks
    overridden = []
    for task in tasks:
        metadata = getattr(task, "metadata", None)
        dataset = getattr(metadata, "dataset", None)
        if not isinstance(dataset, dict):
            continue
        if dataset.get("path") != "dwzhu/LongEmbed":
            continue
        dataset["path"] = str(local_dir)
        dataset.pop("revision", None)
        overridden.append(getattr(metadata, "name", str(task)))
    return overridden


def load_bright_excluded_ids(selected_tasks, subset_name: str):
    task_names = iter_task_names(selected_tasks)
    bright_tasks = [name for name in task_names if "Bright" in name]
    if not bright_tasks:
        return None

    try:
        from datasets import load_dataset
    except Exception:
        return None

    first_task = bright_tasks[0]
    resolved_subset = BRIGHT_SUBSET_MAPPING.get(first_task, subset_name)
    try:
        data_examples = load_dataset("xlangai/BRIGHT", "examples")[resolved_subset]
        return data_examples["excluded_ids"]
    except Exception:
        return None


def sanitize_revision_part(value: str) -> str:
    cleaned = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    sanitized = "".join(cleaned).strip("._")
    return sanitized or "unknown"


def build_model_revision(args) -> str:
    model_ref = args.peft_model_name_or_path or args.base_model_name_or_path
    model_leaf = Path(str(model_ref).rstrip("/")).name or "model"
    parts = [
        sanitize_revision_part(model_leaf),
        f"sim-{sanitize_revision_part(args.similarity_mode)}",
        f"doc-{sanitize_revision_part(args.doc_encoding_mode)}",
    ]
    return "__".join(parts)


def build_model_meta(
    ModelMeta,
    model_name: str,
    use_instructions: bool,
    revision: str,
):
    canonical_name = str(model_name).strip().replace("\\", "/").strip("/")
    canonical_name = canonical_name.replace(" ", "_")
    if not canonical_name:
        canonical_name = "unnamed_model"
    if "/" not in canonical_name:
        canonical_name = f"local/{canonical_name}"
    else:
        canonical_name = f"local/{canonical_name.replace('/', '__')}"

    fallback = SimpleNamespace(
        name=canonical_name,
        revision=revision,
        framework=["LLM2Vec", "PyTorch"],
        use_instructions=use_instructions,
    )
    if ModelMeta is None:
        return fallback
    if hasattr(ModelMeta, "create_empty"):
        try:
            return ModelMeta.create_empty(
                overwrites={
                    "name": str(model_name),
                    "revision": revision,
                    "framework": ["LLM2Vec", "PyTorch"],
                    "use_instructions": use_instructions,
                }
            )
        except Exception:
            return fallback
    try:
        return ModelMeta(
            loader=None,
            loader_kwargs={},
            name=canonical_name,
            revision=revision,
            release_date=None,
            languages=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=None,
            public_training_code=None,
            public_training_data=None,
            framework=["LLM2Vec", "PyTorch"],
            reference=None,
            similarity_fn_name="cosine",
            use_instructions=use_instructions,
            training_datasets=None,
            adapted_from=None,
            superseded_by=None,
            modalities=["text"],
            is_cross_encoder=None,
            citation=None,
            contacts=None,
        )
    except Exception:
        return fallback


class OfficialMTEBWrapper:
    def __init__(
        self,
        model,
        task_to_instructions: dict[str, str] | None = None,
        mteb_model_meta=None,
        default_instruction_resolver=None,
        similarity_mode: str = "cosine",
        doc_chunk_size: int = 0,
        doc_chunk_overlap: int = 0,
        doc_agg_mode: str = "mean",
        doc_top_k: int = 3,
        doc_softmax_temperature: float = 1.0,
        disable_chunking_for_short_docs: bool = False,
        dice_position_mode: str = "reset",
        dice_chunk_input_mode: str = "text",
    ):
        self.model = model
        self.task_to_instructions = task_to_instructions or {}
        self.mteb_model_meta = mteb_model_meta
        self.model_meta = mteb_model_meta
        self.default_instruction_resolver = default_instruction_resolver
        self.similarity_mode = similarity_mode
        self.doc_chunk_size = max(0, int(doc_chunk_size))
        self.doc_chunk_overlap = max(0, int(doc_chunk_overlap))
        self.doc_agg_mode = doc_agg_mode
        self.doc_top_k = max(1, int(doc_top_k))
        self.doc_softmax_temperature = max(float(doc_softmax_temperature), 1e-6)
        self.disable_chunking_for_short_docs = disable_chunking_for_short_docs
        self.dice_position_mode = dice_position_mode
        self.dice_chunk_input_mode = dice_chunk_input_mode

        if self.doc_chunk_overlap >= self.doc_chunk_size and self.doc_chunk_size > 0:
            raise ValueError("doc_chunk_overlap must be smaller than doc_chunk_size")

        if self.doc_chunk_size > 0 and self.doc_chunk_size > self.model.doc_max_length:
            print(
                "Warning: doc_chunk_size is larger than model.doc_max_length "
                f"({self.doc_chunk_size} > {self.model.doc_max_length}). "
                "Each chunk will still be truncated by the model."
            )

    @staticmethod
    def _combine_corpus_text(title: str, text: str) -> str:
        return " ".join(part for part in [title, text] if part).strip()

    @staticmethod
    def _normalize_prompt_type(prompt_type):
        return getattr(prompt_type, "value", prompt_type)

    @staticmethod
    def _is_document_prompt(prompt_type) -> bool:
        return prompt_type in {"passage", "document"}

    @staticmethod
    def _format_instruction(instruction: str) -> str:
        instruction = instruction or ""
        if instruction and instruction[-1] != ":":
            instruction = instruction.strip(".") + ":"
        return instruction

    @staticmethod
    def _use_safe_multi_process(task_name: str | None) -> bool:
        return task_name == "BrightLeetcodeRetrieval"

    def _extract_texts(self, inputs, prompt_type=None):
        prompt_type = self._normalize_prompt_type(prompt_type)
        if isinstance(inputs, dict):
            if self._is_document_prompt(prompt_type):
                titles = inputs.get("title", [""] * len(inputs["text"]))
                return [
                    self._combine_corpus_text(title, text)
                    for title, text in zip(titles, inputs["text"])
                ]
            if "text" in inputs:
                return list(inputs["text"])
            raise TypeError(f"Unsupported input dict keys: {list(inputs.keys())}")

        if isinstance(inputs, (list, tuple)):
            if not inputs:
                return []
            if isinstance(inputs[0], str):
                return list(inputs)
            if isinstance(inputs[0], dict):
                if self._is_document_prompt(prompt_type):
                    return [
                        self._combine_corpus_text(item.get("title", ""), item.get("text", ""))
                        for item in inputs
                    ]
                return [item.get("text", "") for item in inputs]

        if isinstance(inputs, Iterable) and not isinstance(inputs, (str, bytes)):
            texts = []
            for batch in inputs:
                texts.extend(self._extract_texts(batch, prompt_type=prompt_type))
            return texts

        raise TypeError(f"Unsupported inputs type: {type(inputs)}")

    def encode(
        self,
        sentences,
        *,
        task_name: str | None = None,
        prompt_type=None,
        prompt_name: str | None = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        task_metadata=None,
        batch_size: int = 32,
        **kwargs,
    ):
        del hf_split, hf_subset, task_metadata
        prompt_name = prompt_name or task_name
        prompt_type = self._normalize_prompt_type(prompt_type)
        instruction = ""
        if prompt_name and prompt_type == "query":
            if prompt_name in self.task_to_instructions:
                instruction = self.task_to_instructions[prompt_name]
            elif self.default_instruction_resolver is not None:
                instruction = self.default_instruction_resolver(prompt_name)
            instruction = self._format_instruction(instruction)

        texts = self._extract_texts(sentences, prompt_type=prompt_type)
        self.model.use_safe_multi_process = self._use_safe_multi_process(prompt_name)
        if self._is_document_prompt(prompt_type):
            kwargs.pop("request_qid", None)
            kwargs.pop("prompt_name", None)
            return self.model.encode_documents_with_chunk_agg(
                texts=texts,
                batch_size=batch_size,
                chunk_size=self.doc_chunk_size,
                chunk_overlap=self.doc_chunk_overlap,
                agg_mode=self.doc_agg_mode,
                top_k=self.doc_top_k,
                softmax_temperature=self.doc_softmax_temperature,
                disable_chunking_for_short_docs=self.disable_chunking_for_short_docs,
                dice_position_mode=self.dice_position_mode,
                dice_chunk_input_mode=self.dice_chunk_input_mode,
            )

        pairs = [[instruction, text] for text in texts]
        kwargs.pop("request_qid", None)
        return self.model.encode(pairs, batch_size=batch_size, **kwargs)

    def encode_queries(self, queries: List[str], **kwargs):
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus, **kwargs):
        current_task_name = kwargs.get("prompt_name") or kwargs.get("task_name")
        self.model.use_safe_multi_process = self._use_safe_multi_process(current_task_name)
        texts = self._extract_texts(corpus, prompt_type="passage")
        batch_size = kwargs.pop("batch_size", 32)
        kwargs.pop("request_qid", None)
        kwargs.pop("prompt_name", None)
        return self.model.encode_documents_with_chunk_agg(
            texts=texts,
            batch_size=batch_size,
            chunk_size=self.doc_chunk_size,
            chunk_overlap=self.doc_chunk_overlap,
            agg_mode=self.doc_agg_mode,
            top_k=self.doc_top_k,
            softmax_temperature=self.doc_softmax_temperature,
            disable_chunking_for_short_docs=self.disable_chunking_for_short_docs,
            dice_position_mode=self.dice_position_mode,
            dice_chunk_input_mode=self.dice_chunk_input_mode,
        )

    def similarity(self, embeddings1, embeddings2):
        tensor1 = torch.as_tensor(embeddings1)
        tensor2 = torch.as_tensor(embeddings2)
        if self.similarity_mode == "cosine":
            tensor1 = F.normalize(tensor1, p=2, dim=-1)
            tensor2 = F.normalize(tensor2, p=2, dim=-1)
        return tensor1 @ tensor2.T

    def similarity_pairwise(self, embeddings1, embeddings2):
        tensor1 = torch.as_tensor(embeddings1)
        tensor2 = torch.as_tensor(embeddings2)
        if self.similarity_mode == "cosine":
            tensor1 = F.normalize(tensor1, p=2, dim=-1)
            tensor2 = F.normalize(tensor2, p=2, dim=-1)
        return (tensor1 * tensor2).sum(dim=-1)


def run_evaluation(
    mteb_module,
    ResultCache,
    model,
    selected_tasks,
    output_dir: Path,
    prediction_dir: Path,
    batch_size: int,
    excluded_ids,
    preproc: bool,
    fast_bright_root: str | None,
):
    encode_kwargs = {"batch_size": batch_size}

    if hasattr(mteb_module, "evaluate"):
        result_cache = (
            ResultCache(cache_path=output_dir) if ResultCache is not None else None
        )
        kwargs = {
            "model": model,
            "tasks": selected_tasks,
            "prediction_folder": prediction_dir,
            "encode_kwargs": encode_kwargs,
        }
        if result_cache is not None:
            kwargs["cache"] = result_cache

        eval_sig = inspect.signature(mteb_module.evaluate)
        supported = eval_sig.parameters
        if "excluded_ids" in supported and excluded_ids is not None:
            kwargs["excluded_ids"] = excluded_ids
        if "preproc" in supported:
            kwargs["preproc"] = preproc
        if "fast_bright_root" in supported and fast_bright_root is not None:
            kwargs["fast_bright_root"] = fast_bright_root

        results = mteb_module.evaluate(**kwargs)
        return results, result_cache

    if hasattr(mteb_module, "MTEB"):
        tasks_arg = selected_tasks.tasks if hasattr(selected_tasks, "tasks") else selected_tasks
        evaluation = mteb_module.MTEB(tasks=tasks_arg)
        run_sig = inspect.signature(evaluation.run)
        kwargs = {
            "model": model,
            "output_folder": str(output_dir),
            "batch_size": batch_size,
        }
        if "save_predictions" in run_sig.parameters:
            kwargs["save_predictions"] = True
        if "top_k" in run_sig.parameters:
            kwargs["top_k"] = 200
        if "excluded_ids" in run_sig.parameters and excluded_ids is not None:
            kwargs["excluded_ids"] = excluded_ids
        if "preproc" in run_sig.parameters:
            kwargs["preproc"] = preproc
        if "fast_bright_root" in run_sig.parameters and fast_bright_root is not None:
            kwargs["fast_bright_root"] = fast_bright_root
        results = evaluation.run(model=model, **{k: v for k, v in kwargs.items() if k != "model"})
        return results, None

    raise RuntimeError("Installed MTEB does not expose `evaluate` or `MTEB.run`.")


def main():
    args = parse_args()
    mteb, installed_version = load_official_mteb()
    patched_cached_lemb_configs = patch_mteb_cached_lemb_configs()
    LLM2Vec = load_llm2vec()
    ModelMeta = load_model_meta_class(mteb)
    ResultCache = load_result_cache_class(mteb)
    task_to_instructions = None
    doc_encoding_config = apply_doc_encoding_overrides(
        resolve_doc_encoding_config(args.doc_encoding_mode),
        args,
    )

    enable_bidirectional = args.base_model_name_or_path not in ["intfloat/e5-mistral-7b-instruct"]

    torch_dtype = (
        args.torch_dtype if args.torch_dtype == "auto" else getattr(torch, args.torch_dtype)
    )
    l2v_model = LLM2Vec.from_pretrained(
        args.base_model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        enable_bidirectional=enable_bidirectional,
        torch_dtype=torch_dtype,
        merge_peft=True,
    )

    model_name = args.peft_model_name_or_path or args.base_model_name_or_path
    model_revision = build_model_revision(args)
    model_meta = build_model_meta(
        ModelMeta=ModelMeta,
        model_name=model_name,
        use_instructions=task_to_instructions is not None,
        revision=model_revision,
    )
    model = OfficialMTEBWrapper(
        model=l2v_model,
        task_to_instructions=task_to_instructions,
        mteb_model_meta=model_meta,
        default_instruction_resolver=default_instruction_resolver,
        similarity_mode=args.similarity_mode,
        doc_chunk_size=doc_encoding_config["doc_chunk_size"],
        doc_chunk_overlap=doc_encoding_config["doc_chunk_overlap"],
        doc_agg_mode=doc_encoding_config["doc_agg_mode"],
        doc_top_k=doc_encoding_config["doc_top_k"],
        doc_softmax_temperature=doc_encoding_config["doc_softmax_temperature"],
        disable_chunking_for_short_docs=doc_encoding_config["disable_chunking_for_short_docs"],
        dice_position_mode=doc_encoding_config["dice_position_mode"],
        dice_chunk_input_mode=doc_encoding_config["dice_chunk_input_mode"],
    )

    selected_tasks = resolve_tasks(mteb, args.preset, args.task_names, args.benchmark_name)
    overridden_local_datasets = override_local_dataset_paths(selected_tasks, args.longembed_local_dir)
    excluded_ids = load_bright_excluded_ids(selected_tasks, "leetcode")
    output_dir, prediction_dir = ensure_output_dirs(Path(args.output_dir))

    print(f"Using official MTEB version: {installed_version}")
    print(f"Evaluating tasks from: {args.preset or args.task_names or args.benchmark_name}")
    print(f"Similarity mode: {args.similarity_mode}")
    print(f"Model revision tag: {model_revision}")
    print(
        "Document encoding: "
        f"mode={args.doc_encoding_mode}, "
        f"chunk_size={doc_encoding_config['doc_chunk_size']}, "
        f"chunk_overlap={doc_encoding_config['doc_chunk_overlap']}, "
        f"agg_mode={doc_encoding_config['doc_agg_mode']}, "
        f"skip_short={doc_encoding_config['disable_chunking_for_short_docs']}, "
        f"dice_position={doc_encoding_config['dice_position_mode']}, "
        f"dice_chunk_input={doc_encoding_config['dice_chunk_input_mode']}"
    )
    if patched_cached_lemb_configs:
        print("Patched cached MTEB LEMB dataset configs: corpus/qrels/queries")
    if overridden_local_datasets:
        print(
            "Overrode LongEmbed dataset paths for: "
            + ", ".join(overridden_local_datasets)
        )
    if excluded_ids is not None:
        print("Loaded BRIGHT excluded_ids")

    results, result_cache = run_evaluation(
        mteb_module=mteb,
        ResultCache=ResultCache,
        model=model,
        selected_tasks=selected_tasks,
        output_dir=output_dir,
        prediction_dir=prediction_dir,
        batch_size=args.batch_size,
        excluded_ids=excluded_ids,
        preproc=False,
        fast_bright_root=None,
    )

    if result_cache is not None:
        print(f"MTEB results cache written under: {result_cache.cache_path}")
    print(f"Predictions/output written under: {output_dir}")

    if args.save_summary_json:
        summary_path = write_summary_json(
            output_dir=output_dir,
            results=results,
            selected_tasks=selected_tasks,
            eval_config={
                "base_model_name_or_path": args.base_model_name_or_path,
                "peft_model_name_or_path": args.peft_model_name_or_path,
                "model_revision": model_revision,
                "preset": args.preset,
                "task_names": args.task_names,
                "similarity_mode": args.similarity_mode,
                "doc_encoding_mode": args.doc_encoding_mode,
                "doc_chunk_size": doc_encoding_config["doc_chunk_size"],
                "doc_chunk_overlap": doc_encoding_config["doc_chunk_overlap"],
                "doc_agg_mode": doc_encoding_config["doc_agg_mode"],
                "doc_top_k": doc_encoding_config["doc_top_k"],
                "doc_softmax_temperature": doc_encoding_config["doc_softmax_temperature"],
                "disable_chunking_for_short_docs": doc_encoding_config["disable_chunking_for_short_docs"],
                "dice_position_mode": doc_encoding_config["dice_position_mode"],
                "dice_chunk_input_mode": doc_encoding_config["dice_chunk_input_mode"],
                "longembed_local_dir": args.longembed_local_dir,
            },
        )
        print(f"Summary JSON written to: {summary_path}")


if __name__ == "__main__":
    main()
