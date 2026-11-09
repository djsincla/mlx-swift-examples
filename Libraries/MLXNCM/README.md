# MLXNCM

Inference for **Next-Concept Modeling** (NCM), from Kumbong et al.,
[*Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*](https://arxiv.org/abs/2610.14162)
(Stanford Hazy Research and Together AI, MIT-licensed code at
[HazyResearch/ncm](https://github.com/HazyResearch/ncm)).

> This library is a dAI addition, not part of upstream mlx-swift-examples.

An NCM model writes coarse to fine. A multiscale residual VQ-VAE represents a sequence as
levels of discrete codes, from a handful at the top to one per token at the bottom, and a
transformer predicts every code of a level at once. Generation takes one forward pass per
level, however long the output.

## Checkpoints

The released PyTorch checkpoints are converted once, in the NCM repository:

```sh
python local/mlx/convert.py --config config/eval/xsum.yaml \
    --checkpoint-dir artifacts/xsum --out artifacts-mlx/xsum
```

which writes `model.safetensors` (only the tensors generation needs) and `config.json`
(the settings, with the values PyTorch derives while building the model already worked out).

## Use

```swift
import MLX
import MLXNCM

let model = try NCMModel(directory: URL(fileURLWithPath: "artifacts-mlx/xsum"))

// prefix: [B, P] token ids, padded with the model's prefix pad token
let codes = try model.generate(prefix: prefix, temperature: 0) { level, z in
    // the running decoder input after each level, coarse to fine
    let draft = argMax(model.decodeLogits(z), axis: -1)
}
let tokens = argMax(model.decodeLogits(model.zHat(codes: codes)), axis: -1)
```

Tokenisation is the caller's: each checkpoint names its pre-tokenizer in `config.json`
(GPT-2, BERT and Qwen3 vocabularies from Hugging Face, a BPE file pair, or Sudoku digits).
Models that do not predict level 0 themselves (TinyStories, OpenWebText) take `level0:`
codes, sampled from the checkpoint's `inference.level0_unigram`. Some evaluation configs also
re-rank decoded tokens with frequency and adjacency penalties; that step runs on the token
logits after `decodeLogits` and is not part of this library.

## How faithful it is

It follows the generation path of the official PyTorch code operation for operation, except
that the prefix's keys and values are computed once and reused at every level. The prefix
attends only to itself, so a full pass recomputes exactly those values each level.

Checked against the official model, greedy decoding:

| Check | Result |
|---|---|
| Sudoku (hard), XSum: codes at every level, Swift vs PyTorch | identical |
| XSum test set, 11,334 summaries, MLX Python port vs PyTorch | identical; ROUGE-1/2/L 34.92 / 12.71 / 27.93 |

The tests in `Tests/MLXNCMTests` repeat the first check given converted checkpoints; see the
comment at the top of `NCMTests.swift`.

## Supported checkpoints

Every option the released checkpoints use: level 0 from learned queries (Sudoku, XSum, LM1B),
from the prefix (WMT14) or passed in (TinyStories, OpenWebText); codebook-tied, shared and
plain output heads; the post-upsample convolution; single and cascaded level-0 upsamplers.
The converter refuses options outside that set rather than converting them wrongly.
