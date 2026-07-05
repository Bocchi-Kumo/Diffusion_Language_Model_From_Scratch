# Diffusion Language Model from Scratch (SEDD-style)

## Project Goal
This project is my implementation of a discrete diffusion language model inspired by the paper  
[**"Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution"**](https://arxiv.org/abs/2310.16834) (arXiv:2310.16834, SEDD).

The goal is to build a diffusion-based language model from scratch, understand the Score Entropy loss, and explore its potential for supervised fine-tuning (SFT)

## What I Have Implemented

### 1. Model Architecture
I started from a standard autoregressive Transformer that I built from scratch previously and converted it into a diffusion language model:

- Removed causal masks in all attention layers. In diffusion models, the model should see the entire sequence (bidirectional attention) because it needs to predict multiple tokens in parallel during denoising.
- Added a **Timestep Embedding** module and integrated it into every transformer block. This allows the model to condition on the current noise level and predict score ratios accordingly.

The model takes `(x_t, t)` as input and outputs raw logits, which are interpreted as **log S_θ** (log of unnormalized scores) during training.

**Code**:
- [`diffusion_lm`](basics/model.py#L250) — main model class
- [`transformer_block_timestep`](basics/model.py#L239) — transformer block with timestep conditioning
- [`TimestepEmbedding`](basics/model.py#L59) — sinusoidal timestep embedding

### 2. Absorbing Forward Process
I implemented the absorbing-state forward diffusion process:
- Given clean tokens `x_0`, we gradually replace tokens with a special `[MASK]` token according to a noise schedule.
- At each timestep `t`, a token remains unchanged with probability `e^{-σ(t)}` and transitions to `[MASK]` with probability `1 - e^{-σ(t)}`.

This produces the noisy sequence `x_t` that is fed into the model.

**Code**: [`forward_diffusion_absorb`](basics/nn_utils.py#L55)

### 3. Score Entropy Loss (Absorbing)
I implemented the **Score Entropy loss** specifically for the absorbing case:
- The loss is only computed on positions where `x_t` is the mask token.
- For each masked position, the model predicts scores for all possible original tokens.
- The loss encourages the model to assign high score to the correct original token and near-zero scores to incorrect ones.
- This follows the denoising score entropy formulation from the paper.

**Code**: [`denoising_score_entropy`](basics/nn_utils.py#L64)

### 4. Parallel Inference Loop
I built a parallel inference (sampling) loop that can unmask **multiple tokens per step**. At each denoising step, the model selects the most confident masked positions and fills them in parallel.

**Code**: [`inference_simple`](basics/inference_simple.py#L8)

## Current Limitations

- The inference strategy uses a fixed number of tokens to unmask per step. An adaptive confidence-based strategy has not been implemented yet.
- Supervised Fine-Tuning (SFT) has not been implemented yet.
- Reinforcement Learning exploration has no idea.

## Planned Next Steps

### 1. Adaptive Confidence-Based Inference Strategy
I have already implemented a parallel unmasking inference loop that can unmask multiple tokens at each denoising step. The next improvement is to make the number of tokens unmasked **adaptive** based on the model's own confidence:

- **Confidence-aware scheduling**: Instead of using a fixed number of tokens to unmask per step, the model should dynamically decide how many tokens to generate based on its prediction confidence.
- **High confidence → more tokens**: When the model is confident about its predictions (e.g., high probability mass on top predictions), it should unmask more tokens in parallel, accelerating generation.
- **Low confidence → fewer tokens**: When the model is uncertain (e.g., flat probability distribution), it should unmask fewer tokens or even skip a step, allowing the diffusion process to provide more gradual refinement.
- This approach could potentially replace or augment traditional noise schedules with a data-driven, model-guided strategy that adapts to the difficulty of each generation step.

### 2. Supervised Fine-Tuning (SFT)
I plan to implement SFT in the following way:
- Concatenate prompt and response into one sequence as `x_0`.
- During the forward diffusion process, **only mask tokens in the response part** (keep prompt tokens visible).
- Compute the Score Entropy loss **only on the masked (response) positions**.
- This way the model learns to generate responses conditioned on the prompt, similar to instruction tuning but in the diffusion framework.

I believe SFT for diffusion LM is conceptually close to pre-training, but the key difference is the selective masking strategy on the target portion.

## Summary
I have built the core components of a discrete diffusion language model from scratch, including the model, absorbing forward process, and Score Entropy loss. The next critical steps are improving the inference loop and implementing SFT to demonstrate the model's practical capability.

This project serves as both a technical exercise to deeply understand the SEDD paper and a foundation for exploring new architectures for diffusion large language models.
