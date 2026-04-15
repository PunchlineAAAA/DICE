import json
import logging
import os
from functools import partial
from typing import Dict, List, Optional, Union
from copy import deepcopy

import numpy as np
import torch
import torch.multiprocessing as mp
from peft import PeftModel
from torch import Tensor, device, nn
from tqdm.autonotebook import tqdm, trange
from transformers import (
    AutoModel,
    AutoConfig,
    PretrainedConfig,
    AutoTokenizer,
    LlamaConfig,
    MistralConfig,
    GemmaConfig,
    Qwen2Config,
)
from dream.modeling_dream import DreamConfig
from .models import (
    MistralBiModel,
    LlamaBiModel,
    GemmaBiModel,
    Qwen2BiModel,
    DreamBiModel,
)

logger = logging.getLogger(__name__)

def _ensure_transformers_layer_type_validation():
    """
    Backward-compatibility shim for remote configs that import
    `layer_type_validation` from `transformers.configuration_utils`.
    Some Transformers versions do not expose this symbol.
    """
    try:
        from transformers import configuration_utils as cu
    except Exception:
        return

    if hasattr(cu, "layer_type_validation"):
        return

    def _layer_type_validation(layer_type):
        return layer_type

    cu.layer_type_validation = _layer_type_validation


def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


class LLM2Vec(nn.Module):
    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        pooling_mode: str = "mean",
        max_length: int = 512,
        doc_max_length: int = 400,
        skip_instruction: bool = True,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.pooling_mode = pooling_mode
        self.skip_instruction = skip_instruction
        self.max_length = max_length
        self.doc_max_length = doc_max_length
        self.config = model.config

    @classmethod
    def _get_model_class(cls, config_class_name, enable_bidirectional):
        if config_class_name == "DreamConfig" and not enable_bidirectional:
            return DreamBiModel
        if not enable_bidirectional:
            return AutoModel
        if config_class_name == "MistralConfig":
            return MistralBiModel
        elif config_class_name == "LlamaConfig":
            return LlamaBiModel
        elif config_class_name == "GemmaConfig":
            return GemmaBiModel
        elif config_class_name == "Qwen2Config":
            return Qwen2BiModel
        elif config_class_name in ["Fast_dLLM_QwenConfig", "FastdLLMQwenConfig"]:
            # Fast-dLLM v2 reuses the Qwen backbone with a custom config class name.
            return Qwen2BiModel
        elif "Qwen" in config_class_name:
            # Fallback for Qwen-compatible remote configs with non-standard class names.
            return Qwen2BiModel
        elif config_class_name == "DreamConfig":
            return DreamBiModel
        else:
            raise ValueError(
                f"{config_class_name} is not supported yet with bidirectional models."
            )

    @classmethod
    def from_pretrained(
        cls,
        base_model_name_or_path,
        peft_model_name_or_path=None,
        merge_peft=False,
        enable_bidirectional=True,
        **kwargs,
    ):
        # pop out encoder args
        keys = ["pooling_mode", "max_length", "doc_max_length", "skip_instruction"]
        encoder_args = {
            key: kwargs.pop(key, None) for key in keys if kwargs.get(key) is not None
        }

        base_is_peft = False
        if base_model_name_or_path=="siyue/LLM2Vec-Qwen2.5-7B-Instruct-mntp":
            base_is_peft = True
            base_peft = "siyue/LLM2Vec-Qwen2.5-7B-Instruct-mntp"
            base_model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
        

        if os.path.isdir(base_model_name_or_path):
            has_hf_config = os.path.exists(f"{base_model_name_or_path}/config.json")
            if not has_hf_config:
                raise ValueError(
                    f"`base_model_name_or_path={base_model_name_or_path}` looks like a local code directory, "
                    "but no HuggingFace `config.json` was found. "
                    "Please pass a model id (e.g. `Efficient-Large-Model/Fast_dLLM_v2_7B`) "
                    "or a local checkpoint snapshot directory that contains `config.json`."
                )

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name_or_path,
                trust_remote_code=True,
                use_fast=True,
            )
        except Exception as e:
            logger.warning(
                "fast tokenizer failed, fallback to slow tokenizer for %s: %s",
                base_model_name_or_path,
                e,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name_or_path,
                trust_remote_code=True,
                use_fast=False,
            )
            
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        _ensure_transformers_layer_type_validation()
        try:
            config = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=True)
        except ImportError as e:
            if "layer_type_validation" in str(e):
                raise ImportError(
                    "Failed to import Fast-dLLM v2 remote config due to Transformers compatibility: "
                    "missing `layer_type_validation`. Please upgrade transformers to a version compatible "
                    "with Efficient-Large-Model/Fast_dLLM_v2_7B, or pin to the model's recommended version."
                ) from e
            raise
        config_class_name = config.__class__.__name__

        model_class = cls._get_model_class(
            config_class_name, enable_bidirectional=enable_bidirectional
        )
        model = model_class.from_pretrained(base_model_name_or_path, **kwargs)

        if os.path.isdir(base_model_name_or_path) and os.path.exists(
            f"{base_model_name_or_path}/config.json"
        ):
            with open(f"{base_model_name_or_path}/config.json", "r") as fIn:
                config_dict = json.load(fIn)
            config = PretrainedConfig.from_dict(config_dict)
            model.config._name_or_path = config._name_or_path

        # For special case where config.json and adapter weights are in the same directory
        if hasattr(model, "peft_config"):
            model = PeftModel.from_pretrained(
                model,
                base_model_name_or_path,
            )
            model = model.merge_and_unload()
            print(f'merged {base_model_name_or_path} peft.')

        if base_is_peft:
            model = PeftModel.from_pretrained(
                model,
                base_peft,
            )
            model = model.merge_and_unload()
            print(f'merged {base_peft} peft.')

        if peft_model_name_or_path is not None:
            print(f'initialize {peft_model_name_or_path} peft.')
            if merge_peft:
                model = PeftModel.from_pretrained(
                    model,
                    peft_model_name_or_path,
                )
                model = model.merge_and_unload()
                print('merged new peft.')
            else:
                model = PeftModel.from_pretrained(
                    model,
                    peft_model_name_or_path,
                    is_trainable=True
                )
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f'Number of trainable parameters: {trainable_params}')

        config = {}
        config_addr = (
            peft_model_name_or_path
            if peft_model_name_or_path is not None
            else base_model_name_or_path
        )
        if os.path.exists(f"{config_addr}/llm2vec_config.json"):
            with open(f"{config_addr}/llm2vec_config.json", "r") as fIn:
                llm2vec_config = json.load(fIn)
            config.update(llm2vec_config)

        if base_model_name_or_path == "intfloat/e5-mistral-7b-instruct":
            llm2vec_config = {
                "pooling_mode": "eos_token",
                "max_length": 4096,
                "doc_max_length": 4096,
                "skip_instruction": False,
            }
            config.update(llm2vec_config)
            print(f'load {base_model_name_or_path} to update llm2vec_config:')
            print(llm2vec_config["pooling_mode"])

        for key, value in encoder_args.items():
            config[key] = value

        return cls(model=model, tokenizer=tokenizer, **config)
    

    def prepare_for_tokenization(self, text):
        if self.model.config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
            text = (
                "<|start_header_id|>user<|end_header_id|>\n\n"
                + text.strip()
                + "<|eot_id|>"
            )
            return text
        if self.model.config._name_or_path in [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf",
        ]:
            text = "[INST] " + text.strip() + " [/INST]"
        if self.model.config._name_or_path in [
            "google/gemma-2-9b-it",
        ]:
            text = "<bos><start_of_turn>user\n" + text.strip() + "<end_of_turn>"
        if self.model.config._name_or_path in [
            "Qwen/Qwen2-1.5B-Instruct",
            "Qwen/Qwen2-7B-Instruct",
        ]:
            text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"
        ##
        if self.model.config._name_or_path in [
            "Dream-org/Dream-v0-Instruct-7B",
            "siyue/Dream_emb",
        ]:
            text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"
        ##
        if self.pooling_mode == "eos_token":
            if self.model.config._name_or_path == "meta-llama/Meta-Llama-3-8B":
                text = text.strip() + "<|end_of_text|>"
            elif isinstance(self.model.config, LlamaConfig) or isinstance(
                self.model.config, MistralConfig
            ):
                text = text.strip() + " </s>"
            elif isinstance(self.model.config, GemmaConfig):
                text = text.strip() + "<eos>"
            elif isinstance(self.model.config, Qwen2Config):
                text = text.strip() + "<|endoftext|>"
            ##
            elif isinstance(self.model.config, DreamConfig):
                text = text.strip() + "<|endoftext|>"
            ##
        return text

    def tokenize(self, texts):
        texts_2 = []
        original_texts = []
        for text in texts:
            t = text.split("!@#$%^&*()")
            texts_2.append(t[1] if len(t) > 1 else "")
            original_texts.append("".join(t))

        original = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        embed_mask = None
        for t_i, t in enumerate(texts_2):
            ids = self.tokenizer(
                [t],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )

            # Create mask of same size as original attention mask
            e_m = torch.zeros_like(original["attention_mask"][t_i])
            
            if len(ids["input_ids"][0]) > 0:
                # Calculate safe length to avoid overflow
                target_length = min(len(ids["input_ids"][0]), len(e_m))
                # Create ones tensor of exactly the right size
                ones_tensor = torch.ones(target_length)
                # Safely assign the values
                e_m[-target_length:] = ones_tensor
                
            if embed_mask is None:
                embed_mask = e_m.unsqueeze(0)
            else:
                embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

        original["embed_mask"] = embed_mask
        return original

    def _skip_instruction(self, sentence_feature):
        assert (
            sentence_feature["attention_mask"].shape
            == sentence_feature["embed_mask"].shape
        )
        sentence_feature["attention_mask"] = sentence_feature["embed_mask"]

    def forward(self, sentence_feature: Dict[str, Tensor]):
        # fix for qwen 
        sentence_feature["input_ids"] = sentence_feature["input_ids"].long()
        #
        embed_mask = None
        if "embed_mask" in sentence_feature:
            embed_mask = sentence_feature.pop("embed_mask")
        reps = self.model(**sentence_feature)
        sentence_feature["embed_mask"] = embed_mask

        return self.get_pooling(sentence_feature, reps.last_hidden_state)

    def get_pooling(self, features, last_hidden_states):  # All models padded from left
        assert (
            self.tokenizer.padding_side == "left"
        ), "Pooling modes are implemented for padding from left."
        if self.skip_instruction:
            self._skip_instruction(features)
        seq_lengths = features["attention_mask"].sum(dim=-1)

        if self.pooling_mode == "mean":
            return torch.stack(
                [
                    last_hidden_states[i, -int(length.item()):, :].mean(dim=0)
                    for i, length in enumerate(seq_lengths)
                ],
                dim=0,
            )
        elif self.pooling_mode == "weighted_mean":
            bs, l, _ = last_hidden_states.shape
            complete_weights = torch.zeros(bs, l, device=last_hidden_states.device)
            for i, seq_l in enumerate(seq_lengths):
                if seq_l > 0:
                    complete_weights[i, -seq_l:] = torch.arange(seq_l) + 1
                    complete_weights[i] /= torch.clamp(
                        complete_weights[i].sum(), min=1e-9
                    )
            return torch.sum(last_hidden_states * complete_weights.unsqueeze(-1), dim=1)
        elif self.pooling_mode == "eos_token" or self.pooling_mode == "last_token":
            return last_hidden_states[:, -1]
        elif self.pooling_mode == "bos_token":
            return last_hidden_states[
                features["input_ids"] == self.tokenizer.bos_token_id
            ]
        else:
            raise ValueError(f"{self.pooling_mode} is not implemented yet.")

    def _convert_to_str(self, instruction, text):
        tokenized_q = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        tokenized_q_length = len(tokenized_q["input_ids"][0])

        while tokenized_q_length > self.doc_max_length:
            reduction_ratio = self.doc_max_length / tokenized_q_length
            reduced_length = int(len(text.split()) * reduction_ratio)
            text = " ".join(text.split()[:reduced_length])
            tokenized_q = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            tokenized_q_length = len(tokenized_q["input_ids"][0])

        return (
            f"{instruction.strip()} !@#$%^&*(){text}"
            if instruction
            else f"!@#$%^&*(){text}"
        )

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        disable_chunking_for_short_docs: bool = False,
    ) -> List[str]:
        if chunk_size <= 0:
            return [text]
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return [text]

        if disable_chunking_for_short_docs and len(token_ids) <= chunk_size:
            return [text]

        step = chunk_size - chunk_overlap
        chunks = []
        for start in range(0, len(token_ids), step):
            chunk_ids = token_ids[start : start + chunk_size]
            if not chunk_ids:
                continue
            chunk_text = self.tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if start + chunk_size >= len(token_ids):
                break
        return chunks or [text]

    def _aggregate_chunk_embeddings(
        self,
        chunk_embeddings: torch.Tensor,
        agg_mode: str = "mean",
        top_k: int = 3,
        softmax_temperature: float = 1.0,
    ) -> torch.Tensor:
        if chunk_embeddings.ndim != 2:
            raise ValueError(
                "chunk_embeddings should be rank-2: [num_chunks, hidden_dim], "
                f"got shape {tuple(chunk_embeddings.shape)}"
            )

        if chunk_embeddings.shape[0] == 1:
            return chunk_embeddings[0]

        if agg_mode == "mean":
            return chunk_embeddings.mean(dim=0)
        if agg_mode == "max":
            return chunk_embeddings.max(dim=0).values

        chunk_norms = torch.linalg.norm(chunk_embeddings, dim=-1)
        if agg_mode == "norm_topk_mean":
            k = min(max(1, int(top_k)), chunk_embeddings.shape[0])
            topk_idx = torch.topk(chunk_norms, k=k, largest=True).indices
            return chunk_embeddings[topk_idx].mean(dim=0)

        if agg_mode == "norm_softmax_mean":
            temperature = max(float(softmax_temperature), 1e-6)
            weights = torch.softmax(chunk_norms / temperature, dim=0)
            return (chunk_embeddings * weights.unsqueeze(-1)).sum(dim=0)

        raise ValueError(f"Unknown agg_mode: {agg_mode}")

    def encode_text_list(
        self,
        texts: List[str],
        batch_size: int = 32,
        instruction: str = "",
        show_progress_bar: bool = False,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        pairs = [[instruction, text] for text in texts]
        embeddings = self.encode(
            pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
            device=device,
        )
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.as_tensor(embeddings)
        return embeddings.to(torch.float32)

    def encode_documents_with_chunk_agg(
        self,
        texts: List[str],
        batch_size: int = 32,
        chunk_size: int = 0,
        chunk_overlap: int = 0,
        agg_mode: str = "mean",
        top_k: int = 3,
        softmax_temperature: float = 1.0,
        disable_chunking_for_short_docs: bool = False,
        instruction: str = "",
        show_progress_bar: bool = False,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        if chunk_size <= 0:
            return self.encode_text_list(
                texts=texts,
                batch_size=batch_size,
                instruction=instruction,
                show_progress_bar=show_progress_bar,
                device=device,
            )

        all_chunks = []
        doc_spans = []
        cursor = 0
        for text in texts:
            chunks = self._chunk_text(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                disable_chunking_for_short_docs=disable_chunking_for_short_docs,
            )
            doc_spans.append((cursor, cursor + len(chunks)))
            cursor += len(chunks)
            all_chunks.extend(chunks)

        chunk_embeddings = self.encode_text_list(
            texts=all_chunks,
            batch_size=batch_size,
            instruction=instruction,
            show_progress_bar=show_progress_bar,
            device=device,
        )

        aggregated_embeddings = []
        for start, end in doc_spans:
            aggregated_embeddings.append(
                self._aggregate_chunk_embeddings(
                    chunk_embeddings[start:end],
                    agg_mode=agg_mode,
                    top_k=top_k,
                    softmax_temperature=softmax_temperature,
                )
            )
        return torch.stack(aggregated_embeddings, dim=0).to(torch.float32)

    def forward_text_list(
        self,
        texts: List[str],
        instruction: str = "",
        device: Optional[str] = None,
    ) -> torch.Tensor:
        if device is None:
            device = str(next(self.parameters()).device)

        combined_texts = [
            self.prepare_for_tokenization(self._convert_to_str(instruction, text))
            for text in texts
        ]
        features = self.tokenize(combined_texts)
        features = batch_to_device(features, device)
        return self.forward(features)

    def forward_documents_with_chunk_agg(
        self,
        texts: List[str],
        chunk_size: int = 0,
        chunk_overlap: int = 0,
        agg_mode: str = "mean",
        top_k: int = 3,
        softmax_temperature: float = 1.0,
        disable_chunking_for_short_docs: bool = False,
        instruction: str = "",
        device: Optional[str] = None,
    ) -> torch.Tensor:
        if device is None:
            device = str(next(self.parameters()).device)

        if chunk_size <= 0:
            return self.forward_text_list(
                texts=texts,
                instruction=instruction,
                device=device,
            )

        all_chunks = []
        doc_spans = []
        cursor = 0
        for text in texts:
            chunks = self._chunk_text(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                disable_chunking_for_short_docs=disable_chunking_for_short_docs,
            )
            doc_spans.append((cursor, cursor + len(chunks)))
            cursor += len(chunks)
            all_chunks.extend(chunks)

        chunk_embeddings = self.forward_text_list(
            texts=all_chunks,
            instruction=instruction,
            device=device,
        )

        aggregated_embeddings = []
        for start, end in doc_spans:
            aggregated_embeddings.append(
                self._aggregate_chunk_embeddings(
                    chunk_embeddings[start:end],
                    agg_mode=agg_mode,
                    top_k=top_k,
                    softmax_temperature=softmax_temperature,
                )
            )
        return torch.stack(aggregated_embeddings, dim=0)

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = False,
        device: Optional[str] = None,
    ):
        """
        Encode a list of sentences to their respective embeddings. The sentences can be a list of strings or a string.
        Args:
            sentences: sentence or sentences to encode.
            batch_size: batch size for turning sentence tokens into embeddings.
            show_progress_bar: whether to show progress bars during encoding steps.
            convert_to_numpy: If true, return numpy arrays instead of torch tensors.
            convert_to_tensor: If true, return torch tensors (default).
            device: torch backend device identifier (e.g., 'cuda', 'cpu','mps' etc.). If not specified,
            the default is to use cuda when available, otherwise cpu. Note that only the choice of 'cuda' supports
            multiprocessing as currently implemented.

        Returns: embeddings of the sentences. Embeddings are detached and always on the CPU (see _encode implementation).

        """
        if isinstance(sentences[0], str) and isinstance(sentences[-1], int):
            sentences = [sentences]
        # required for MEDI version of MTEB
        if isinstance(sentences[0], str):
            sentences = [[""] + [sentence] for sentence in sentences]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        concatenated_input_texts = []
        for sentence in sentences:
            assert isinstance(sentence[0], str)
            assert isinstance(sentence[1], str)
            concatenated_input_texts.append(
                self._convert_to_str(sentence[0], sentence[1])
            )
        sentences = concatenated_input_texts

        self.eval()

        if convert_to_tensor:
            convert_to_numpy = False

        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        all_embeddings = []
        use_local_device_only = (
            torch.distributed.is_initialized()
            or (isinstance(device, str) and device.startswith("cuda:"))
            or (device != "cuda")
        )

        if torch.cuda.device_count() <= 1 or use_local_device_only:
            # This branch also support mps devices
            self.to(device)
            for start_index in trange(
                0,
                len(sentences),
                batch_size,
                desc="Batches",
                disable=not show_progress_bar,
            ):
                sentences_batch = sentences_sorted[
                    start_index : start_index + batch_size
                ]
                embeddings = self._encode(
                    sentences_batch, device=device, convert_to_numpy=convert_to_numpy
                )
                all_embeddings.append(embeddings)
        else:
            num_proc = torch.cuda.device_count()
            cuda_compatible_multiprocess = mp.get_context("spawn")
            with cuda_compatible_multiprocess.Pool(num_proc) as p:
                sentences_batches = [
                    sentences_sorted[start_index : start_index + batch_size]
                    for start_index in range(0, len(sentences), batch_size)
                ]

                progress_bar = tqdm(
                    total=len(sentences_batches),
                    desc="Batches",
                    disable=not show_progress_bar,
                )
                results = []

                def update(*args):
                    progress_bar.update()

                for batch in sentences_batches:
                    results.append(
                        p.apply_async(
                            self._encode,
                            args=(batch, None, convert_to_numpy, True),
                            callback=update,
                        )
                    )

                all_embeddings = [result.get() for result in results]
                progress_bar.close()

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]
        all_embeddings = all_embeddings.to(torch.float32)
        if convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings

    def save(self, output_path, merge_before_save=False, save_config=True):
        if merge_before_save and isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
            # Fixes the issue of saving - https://huggingface.co/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse/discussions/1
            if hasattr(self.model, "_hf_peft_config_loaded"):
                self.model._hf_peft_config_loaded = False

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        llm2vec_config = {
            "pooling_mode": self.pooling_mode,
            "max_length": self.max_length,
            "doc_max_length": self.doc_max_length,
            "skip_instruction": self.skip_instruction,
        }

        if save_config:
            os.makedirs(output_path, exist_ok=True)
            with open(f"{output_path}/llm2vec_config.json", "w") as fOut:
                json.dump(llm2vec_config, fOut, indent=4)

    def _encode(
        self,
        sentences_batch,
        device: Optional[str] = None,
        convert_to_numpy: bool = False,
        multiprocessing=False,
    ):
        if multiprocessing:
            # multiprocessing only supports CUDA devices at this time, so we ignore the value of device
            # and use cuda:rank for the device
            rank = mp.current_process()._identity[0]
            if device is None and torch.cuda.is_available():
                device = f"cuda:{rank % torch.cuda.device_count()}"

        self.to(device)
        features = self.tokenize(
            [self.prepare_for_tokenization(sentence) for sentence in sentences_batch]
        )
        features = batch_to_device(features, device)

        with torch.no_grad():
            embeddings = self.forward(features)
            if self.model.config._name_or_path == "intfloat/e5-mistral-7b-instruct":
                import torch.nn.functional as F
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            embeddings = embeddings.detach()
            embeddings = embeddings.cpu()

        return embeddings

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either a string (which means a single text)
        a list of ints (which means a single tokenized text), or a tuple of list of ints
        (representing several text inputs to the model).
        """
        if (
            isinstance(text, str)
            or (isinstance(text, list) and isinstance(text[0], int))
            or len(text) == 0
        ):  # Single text, list of ints, or empty
            return len(text)
        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        else:
            return sum([len(t) for t in text])

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        return self.model.resize_token_embeddings(
            new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
