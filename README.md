# DICE-Embedding

DICE-Embedding is a long-document retrieval codebase centered on **DICE**: **D**ocument **I**nference via **C**hunk **E**vidence.

The main idea is simple: instead of encoding a long document into one vector in a single pass, DICE splits the document into chunks, encodes chunks independently, and aggregates chunk evidence back into a single document representation while keeping the standard one-query-one-document retrieval interface.

This repository builds on top of `LLM2Vec`-style encoder wrapping and includes support for multiple backbones, including Dream, Llama, Mistral, Gemma, and Qwen.

## Repository Layout

- `llm2vec/`: core model wrapper and document encoding logic
- `experiments/`: training and evaluation entrypoints
- `dream/`: Dream model integration
- `train_configs/`: supervised, SimCSE, and MNTP training configs
- `test_configs/`: evaluation configs
- `ReasonAug/`: reasoning-retrieval data utilities
- `assets/`: figures and static assets
- `examples/`: small usage examples

The two main files for DICE-style long-document evaluation are:

- `llm2vec/llm2vec.py`
- `experiments/mteb_eval_v2.py`

## Installation

Create an environment and install dependencies:

```bash
conda create -n dice-embedding python=3.10
conda activate dice-embedding
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

Notes:

- `flash-attn` may require `ninja` and a supported NVIDIA GPU.
- Some evaluation workflows also require local benchmark or dataset caches.

## What DICE Changes

DICE changes only the **document encoding path**.

- Query encoding stays unchanged.
- Documents can be chunked by text or token ids.
- Chunk position handling supports both local reset and absolute-offset variants.
- Chunk embeddings are aggregated back into a single document vector for standard dense retrieval.

This makes DICE easy to test against existing dense retrieval pipelines without changing the retrieval interface.

## Evaluation

The main evaluation entrypoint is:

```bash
python experiments/mteb_eval_v2.py --help
```

Typical controls include:

- backbone model paths
- chunk size / overlap
- DICE position mode
- DICE chunk input mode
- task selection for LongEmbed, FollowIR, BRIGHT, and related evaluation flows

Example:

```bash
python experiments/mteb_eval_v2.py \
  --base_model_name_or_path <base_model> \
  --peft_model_name_or_path <peft_model> \
  --task_name NarrativeQARetrieval \
  --chunk_method chunk \
  --doc_chunk_size 1024 \
  --doc_chunk_overlap 0 \
  --pooling_mode mean \
  --dice_position_mode reset \
  --dice_chunk_input_mode token_ids
```

## Training

The main supervised training entrypoint is:

```bash
torchrun --nproc_per_node=4 experiments/run_supervised.py train_configs/supervised/<config>.json
```

You can adapt configs under `train_configs/` for Dream, Llama, Mistral, Qwen, and related settings.

## Acknowledgment

This codebase builds on ideas and implementations from:

- [LLM2Vec](https://github.com/McGill-NLP/llm2vec/)
- [Dream](https://github.com/HKUNLP/Dream)
- [DiffEmbed](https://github.com/siyue-zhang/DiffEmbed)
