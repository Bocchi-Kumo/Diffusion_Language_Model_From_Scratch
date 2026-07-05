import torch
import torch.nn.functional as F

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    '''
    Computing the softmax for the input tensor, choose the largest number as C(constant)
    '''
    c = x.max(dim=dim, keepdim=True).values
    exp_v = torch.exp(x - c)
    return exp_v / exp_v.sum(dim=dim, keepdim=True)

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    '''
    Args:
    inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
        unnormalized logit of jth class for the ith example.
    targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
        Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    '''
    inputs = inputs - inputs.max(dim=-1, keepdim=True).values
    # log_sum_exp = torch.log(torch.exp(inputs).sum(dim=-1))
    # target_logits = inputs[torch.arange(inputs.shape[0]), targets]
    # loss = -target_logits + log_sum_exp # -log(exp(correct_logit) / sum(exp(all_logits)))
    loss = (-inputs[torch.arange(inputs.shape[0]), targets] + torch.log(torch.exp(inputs).sum(dim=-1)))
    return loss.mean()

def gradient_clipping(parameters, max_l2_norm=1.0, eps=1e-6) -> None:
    '''
    Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    '''
    grads = []
    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    if not grads:
        return
    g = torch.cat(grads)
    l2_norm = torch.norm(g, p=2)
    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for param in parameters:
            if param.grad is not None:
                param.grad.mul_(scale)
    return

def forward_diffusion_absorb(x_0, t, mask_token_id):
    sigma = 1/(1-t)
    bar_sigma = -torch.log(1-t)
    a = torch.exp(-bar_sigma) # prob_keep

    random_mask = torch.rand_like(x_0.float()) > a
    x_t = torch.where(random_mask, mask_token_id, x_0)
    return x_t, sigma, bar_sigma, a

def denoising_score_entropy(
    log_scores: torch.Tensor,      # [B, L, V]  model output = log(s_y)
    x0: torch.Tensor,              # [B, L]
    xt: torch.Tensor,              # [B, L]
    mask_token_id: int,
    sigma: torch.Tensor,           # [B, 1] or broadcastable
    eps: float = 1e-8              # small eps for stability only
) -> torch.Tensor:
    B, L, V = log_scores.shape
    device = log_scores.device

    is_masked = (xt == mask_token_id)
    if not is_masked.any():
        return torch.zeros((), device=device, dtype=log_scores.dtype)

    # --- 1. Stable ratio computation (avoid 0/0 or huge values) ---
    esigm1 = torch.where(
        sigma < 0.5,
        torch.expm1(sigma),
        torch.exp(sigma) - 1
    )
    # Clamp to avoid inf when sigma≈0 and to avoid underflow when sigma large
    esigm1 = esigm1.clamp(min=eps)
    ratio = (1.0 / esigm1).expand_as(xt).clamp(min=eps, max=1e8)   # [B, L]

    # --- 2. Stable const_term: r*(log(r)-1) = xlogy(r,r) - r ---
    # This is the key fix. torch.xlogy(r, r) safely gives 0 when r→0
    ratio_masked = ratio[is_masked]
    r_log_r = torch.xlogy(ratio_masked, ratio_masked)   # stable r * log(r)
    const_term = r_log_r - ratio_masked

    # --- 3. Gather log-score of correct token (already log(s)) ---
    log_s_correct = log_scores[is_masked].gather(
        1, x0[is_masked].unsqueeze(1)
    ).squeeze(1)

    # Optional but recommended: clamp extreme log-scores to prevent inf in exp()
    log_s_correct = log_s_correct.clamp(min=-100.0, max=100.0)

    # --- 4. pos_term: sum s_y over non-mask tokens (stable exp) ---
    non_mask_mask = torch.arange(V, device=device) != mask_token_id
    log_scores_masked = log_scores[is_masked][:, non_mask_mask].clamp(min=-100.0, max=100.0)
    pos_term = log_scores_masked.exp().sum(dim=1)

    # --- 5. neg_term (uses log-score directly, as in official code) ---
    neg_term = ratio_masked * log_s_correct

    # --- 6. Final entropy (can still be large, but not NaN) ---
    entropy = pos_term - neg_term + const_term

    # Safety net (optional but useful while debugging)
    entropy = torch.nan_to_num(entropy, nan=0.0, posinf=1e9, neginf=-1e9)

    # --- 7. Weight by dsigma and reduce ---
    sigma_masked = sigma.expand_as(xt)[is_masked]
    # Make sure dsigma > 0 (your noise schedule should guarantee this)
    sigma_masked = sigma_masked.clamp(min=eps)

    loss = (entropy * sigma_masked).mean()
    return loss