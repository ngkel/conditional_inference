# Conditional Inference

This repository is a playground for consolidating ideas from [Chapter 7 — Inference with Low-Dimensional Distribution](https://ma-lab-berkeley.github.io/deep-representation-learning-book/Ch7.html) of *Principles and Practice of Deep Representation Learning*.

The main artifact is [`experiments.ipynb`](experiments.ipynb). It connects score-based diffusion, classifier-free guidance (CFG), and Gaussian mixture models (GMMs) to Transformer-style architectures—showing how multi-head cross attention can serve as a functional form for both unconditional and conditional denoisers, and why that resemblance matters for modern inference.

## Motivation

Almost all machine learning applications can be framed as an inverse problem:

$$y = h(x) + w$$

where $h(\cdot)$ is a measurement of latent $x$ and $w$ is noise (or corruption). The goal is to recover a plausible $\hat{x}$ consistent with the observation $y$.

In diffusion-based inference, sampling reverses a forward noising process. The score function $\nabla_{x_t} \log p_t(x_t)$ is typically unknown, so denoising is often done via the posterior mean $\mathbb{E}[x_0 \mid x_t]$ (Tweedie's formula links the two). When only paired data $(x, y)$ are available, **conditional denoising** is required.

**Classifier-free guidance (CFG)** blends unconditional and conditional denoisers:

$$\bar{\boldsymbol{x}}_\theta^{\text{CFG}}(t, \boldsymbol{x}_t, y)
  = (1-\gamma)\,\bar{\boldsymbol{x}}_\theta(t, \boldsymbol{x}_t, \varnothing)
  + \gamma\,\bar{\boldsymbol{x}}_\theta(t, \boldsymbol{x}_t, y)$$

with guidance scale $\gamma > 1$.

## Core idea: GMM denoisers and a unified operator

The notebook assumes data drawn from a mixture of low-rank Gaussians:

$$x \sim \frac{1}{K}\sum_{k=1}^{K} \mathcal{N}(0, U_k U_k^{\top})$$

For this GMM, closed-form expressions are derived for:

- the **unconditional** denoiser $\bar{\boldsymbol{x}}_\theta(t, \boldsymbol{x}_t, \varnothing)$ (softmax weighting over components),
- the **conditional** denoiser $\bar{\boldsymbol{x}}_\theta(t, \boldsymbol{x}_t, y)$ (projection onto component $y$),
- and the **CFG** denoiser combining both.

A key step is introducing label embeddings $\boldsymbol{v}_k$ that live in the same space as $x_t$, under a distinguishability assumption on subspaces $U_k$. This yields a **single operator** $(x_t, \boldsymbol{v}) \mapsto \hat{x}$ that recovers:

- the unconditional denoiser when $\boldsymbol{v} = x_t$,
- the class-conditional denoiser when $\boldsymbol{v} = \boldsymbol{v}_y$.

Because any distribution over $x$ can be approximated arbitrarily well by a Gaussian mixture, this GMM analysis is used as a tractable lens on why Transformer-like architectures are effective: if learned representations are (approximately) Gaussian in a suitable space, attention-based denoisers are natural.

## Connection to multi-head cross attention

The notebook compares the GMM conditional denoiser to **multi-head cross attention (MHCA)**. With a single query token, the softmax-over-subspaces structure of the GMM denoiser closely resembles cross attention: queries from the noisy state, keys/values from the condition embedding, and attention weights selecting the right subspace. CFG's $\gamma$ then steers denoising toward the desired class—analogous to guidance in conditional generation.

The same compression viewpoint appears in iterative denoising and in **multi-token subspace attention (MSSA)** for local token compression toward a mixture-of-subspaces structure.

## What the notebook contains

| Section | Content |
|--------|---------|
| **Prerequisites** | Score-based diffusion (forward SDE, reverse ODE, Tweedie's formula) |
| **Conditional inference** | Bayes-optimal conditional denoiser when $p(x)$ is known |
| **CFG for GMM** | Closed-form unconditional, conditional, and CFG denoisers; unified operator |
| **Synthetic 3D GMM** | Train `WeightedEncoderNet` (3 linear encoders + label embedding), sample, visualize |
| **Conditional denoiser** | CFG sampling experiments on the trained model |
| **MHCA vs GMM denoiser** | Side-by-side formulas and interpretation (compression, subspace selection) |
| **FashionMNIST GMM** | Per-class truncated SVD (means $\mu_k$, bases $U_k$, singular values); diffusion on flattened images |
| **Interpretable RAE** | Variance-preserving process, forward diffusion, class-conditioned denoising demos |
| **Open questions** | Memorization vs generalization, causality, cross attention vs timestep conditioning, world models |

## Getting started

Requires Python ≥ 3.10. Dependencies are managed with [uv](https://github.com/astral-sh/uv) (`pyproject.toml`).

```bash
# install dependencies
uv sync

# optional: Jupyter kernel for the notebook
uv sync --group dev

# open the notebook
jupyter notebook experiments.ipynb
```

Main dependencies: PyTorch, torchvision, NumPy, Matplotlib.

## Reference

- Ma, Y., et al. [*Principles and Practice of Deep Representation Learning*](https://ma-lab-berkeley.github.io/deep-representation-learning-book/) — [Chapter 7](https://ma-lab-berkeley.github.io/deep-representation-learning-book/Ch7.html)
