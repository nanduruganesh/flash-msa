# Flash-MSA
Flash-MSA is written in CuTeDSL for Hopper and Blackwell GPUs (eg H100, B200) on CUDA 13.

These kernels implement training for the MiniMax Sparse Attention paper:
https://arxiv.org/abs/2606.13392

Briefly, MSA is a style of sparse attention fitted to GQA that uses a small proxy attention layer to select blocks of keys to provide to the main attention layer. This offers a massive speedup to inference by slashing the memory-bandwidth bottleneck of loading the full KV cache from HBM.

The proxy heads are trained via a KL-divergence loss between the main attention layer's attention scores over the sparsely selected blocks. The proxy heads are assigned groups of main attention heads to select keys for & average scores over for KL-teaching.

This library also includes MSA warmup kernels, which run the main attention densely and train the proxy attention on the full sequence.

More information is included in the [blog post](https://nanduruganesh.github.io/flash-msa).

# Installation

flash-msa depends on FA3/4 from [flash-attn](https://github.com/dao-ailab/flash-attention). Try to configure your CUDA/Python/Torch versions to match one of the flash-attn wheels for a fast installation, but if you must build from source, set `MAX_JOBS=<max jobs>` to avoid `pip install flash-msa[attn]` bricking your CPU.

You will also need Python headers, e.g. `apt-get install python3.12-dev`, for whichever python version you are using.

```
uv pip install flash-msa
```

From source:

```
python setup.py install
```
or
```
uv pip install -e . --no-build-isolation
```
# Usage
```
from flash_msa import flash_msa_func
attn_out, kl_loss = flash_msa_func(Q_proxy, K_proxy, Q, K, V, top_k, head_dim ** -0.5)
```
or
```
from flash_msa import flash_msa_warmup_func
attn_out, kl_loss = flash_msa_warmup_func(Q_proxy, K_proxy, Q, K, V, top_k, head_dim ** -0.5)
```

Note that kl_loss in the forward is just a torch.zeros placeholder, but after adding it to the main model loss, calling backward() will activate the on-the-fly gradient calcs equivalent to the actual proxy KL loss signal.

# Caveats

1. No varlen/doc-masking support yet (highest priority)
2. Flash-MSA only supports headdims 128, block size 128.
3. Flash-MSA does not currently return fully materialized KL div. loss term in the fwd/bwd (see [blog](https://nanduruganesh.github.io/flash-msa) for explanation).
4. No support for quantized training (fp8, nvfp4, mxfp4).
5. No support for attn temps / oai-style softmax bias.
6. Proxy Q is grouped by Main KV so Q_p >= KV heads for now.

These are not ridiculous to implement though so if there is demand or if someone makes a PR, I will update the repo to include these features.

# Testing

Test sparse MSA correctness against an eager implementation of MSA: `python tests/test_eager_match.py [args]`

Test warmup MSA correctness against an eager implementation of MSA: `python tests/test_warmup_eager_match.py [args]`

# Training

An MSA training example is implemented in this [Megatron-LM fork](https://github.com/nanduruganesh/Megatron-LM). 

Notably, you must add the kl_loss returned by MSA kernels to the model's main CE loss before backward to train the proxy attention. The kl_loss is currently treated as a torch.zeros` placeholder and calculated on-the-fly in the backward, so logging the kl_loss will not reflect how proxy training is actually going. Some solutions to get some signal on proxy training are logging grad/update norms of proxy weights, or patching the forward kernel to calculate and accumulate KL div, but only doing this once every n steps to amortize how slow this would make the forward.

In general if you are going to train with this it is highly recommended to follow tips from [the paper](https://arxiv.org/abs/2606.13392), use MSA warmup before turning on MSA sparse training, and replicate any transformations to the main attention queries and keys (RoPE, QK norm, QK clip, etc) to the proxy queries and keys to improve proxy convergence.

# Inference
See MiniMax's [official repo](https://github.com/MiniMax-AI/MSA) for MSA inference kernels.
