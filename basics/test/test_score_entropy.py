import torch
import torch.nn.functional as F
from basics.nn_utils import denoising_score_entropy, forward_diffusion_absorb, softmax

def denoising_score_entropy_naive(
        logits:torch.Tensor, # [batch_size, seq_length, vocab_size]
        x_0: torch.Tensor, # [batch_size, seq_length]
        x_t: torch.Tensor,  # [batch_size, seq_length]
        mask_token_id: int,
        sigma:torch.Tensor, # [batch_size, 1]
        eps: float = 1e-8
) -> torch.Tensor:
    batch_size, seq_length, vocab_size = logits.shape
    logits_clamped = logits.clamp(min=-1000.0, max=1000.0)
    total_loss = 0
    count = 0

    for b in range(batch_size):
        for i in range(seq_length):
            xt_i = x_t[b, i].item()
            if xt_i != mask_token_id:
                # We only calculate loss on xt == mask
                continue

            x0_i = x_0[b, i].item()

            sig = sigma[b, 0].item()
            # prob_keep = (e^(-sigma)) / (1-e(-sigma)) = (1) / (e(sigma) - 1)
            esigma_1 = max((torch.exp(torch.tensor(sig)).item() - 1), eps) # e^(sigma) - 1
            ratio = 1.0 / (esigma_1)
            ratio = max(min(ratio, 1e9), eps)

            pos_term = 0.0
            for y in range(vocab_size):
                if y == mask_token_id:
                    continue
                # S_θ = e^(log_logits)
                pos_term += torch.exp(logits_clamped[b, i, y]) # S_θ()
            
            # neg_term = ratio * logits[x0]  (logits used directly as log-scores)
            neg_term = ratio * logits_clamped[b, i, x0_i].item() # We only add those Y == X_0 with ratio. The other all ratio is 0 so we skip them
            K = ratio * (torch.log(torch.tensor(ratio)) - 1)
            
            # ratio = P(y|x0) / P(xt|x0)
            # K(a) = a * (log(a) - 1)
            # loss = (S_θ - ratio * log(S_θ) + K)
            total_loss += (pos_term - neg_term + K) * sigma[b, 0].item()
            count += 1
        
    if count == 0:
        return torch.tensor(0.0, device=logits.device)
    return total_loss / count

def run_tests():
    torch.manual_seed(42)
    
    test_configs = [
        # (vocab_size, mask_token_id, seq_length, batch_size, num_timesteps, t_range)
        {"vocab": 32,  "mask_id": 0,  "seq": 64,  "batch": 2,  "T": 10, "t_low": 1, "t_high": 10},
        {"vocab": 100, "mask_id": 99, "seq": 128, "batch": 4,  "T": 100,"t_low": 1, "t_high": 100},
        {"vocab": 16,  "mask_id": 0,  "seq": 32,  "batch": 1,  "T": 10, "t_low": 5, "t_high": 6},
        {"vocab": 50,  "mask_id": 25, "seq": 64,  "batch": 8,  "T": 50, "t_low": 1, "t_high": 50},
        {"vocab": 256,  "mask_id": 255, "seq": 128,  "batch": 16,  "T": 1000, "t_low": 1, "t_high": 1000},
    ]
    
    all_passed = True
    
    for idx, cfg in enumerate(test_configs):
        V = cfg["vocab"]
        mask_id = cfg["mask_id"]
        L = cfg["seq"]
        B = cfg["batch"]
        T = cfg["T"]
        
        t = torch.randint(low=cfg["t_low"], high=cfg["t_high"] + 1, size=[B, 1])
        t_norm = t.float() / T
        sigma = 1.0 / (1.0 - t_norm)
        bar_sigma = -torch.log(1.0 - t_norm)
        a = torch.exp(-bar_sigma)

        x_0 = torch.randint(low=0, high=V, size=[B, L])

        x_0 = torch.where(x_0 == mask_id, (mask_id + 1) % V, x_0)
        
        random_mask = torch.rand(B, L) > a 
        x_t = torch.where(random_mask, torch.tensor(mask_id), x_0)
        
        logits = torch.randn(B, L, V)
        
        loss_naive = denoising_score_entropy_naive(
            logits, x_0, x_t, mask_id, sigma,
        )
        loss_vec = denoising_score_entropy(
            logits, x_0, x_t, mask_id, sigma
        )
        
        diff = abs(loss_naive.item() - loss_vec.item())
        rel_diff = diff / (abs(loss_naive.item()) + 1e-8)
        passed = diff < 1e-3
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"Test {idx+1}: V={V:>3}, L={L:>3}, B={B}, "
              f"naive={loss_naive.item():.6f}, vec={loss_vec.item():.6f}, "
              f"diff={diff:.2e}, rel={rel_diff:.2e} → {status}")
        
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 Pass al the tests")
    else:
        print("⚠️  Fail to pass the tests")
    print("=" * 50)

if __name__ == "__main__":
    run_tests()