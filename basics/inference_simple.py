import torch
from basics.helper_function import get_tokenizer
from basics.run_scripts.filepath import TS_MODEL_PATH
from basics.tokenizer import tokenizer
from tqdm import tqdm

@torch.no_grad()
def inference_simple(
        model,
        tokenizer:tokenizer,
        device:str,
        prompt:str,
        seq_length:int,
        temperature:float = 1.0,
        top_k:int = 1,  # Using this to control how many tokens would be unmasker per loop
):
    model.eval()
    mask_token_id = tokenizer.encode("<|mask|>")[0]
    encoded_text = tokenizer.encode(prompt)
    mask_token_length = seq_length - len(encoded_text)
    # Mask all the positions with mask id except the prompt positons.
    x = encoded_text + [mask_token_id for _ in range(mask_token_length)]
    x = torch.tensor([x], dtype=torch.long, device=device)  # [batch_size, seq_length]. 

    num_timesteps = (mask_token_length + top_k - 1) // top_k
    
    for current_t in (reversed(range(num_timesteps))):
        t = torch.tensor([[float(current_t) / float(num_timesteps)]], dtype=torch.float32, device=device)
        with torch.no_grad():
            log_logits = model(x, t) # [batch_size, seq_length, vocab_size]
        scores = torch.exp(log_logits)

        # P(y|xt == mask, t) = S^(1/temperature) / S^(1/temperature).sum() where y != mask
        p_y_xt = scores ** (1.0 / temperature)
        p_y_xt[:, :, mask_token_id] = 0.0
        p_y_xt = p_y_xt / p_y_xt.sum(dim=-1, keepdim=True)
        p_y_xt = p_y_xt.squeeze(0)  # [seq_length, vocab_size]

        top_probs, top_token_ids = torch.topk(p_y_xt, k=top_k, dim=-1)

        # Find positions that are STILL masked
        still_masked = (x[0] == mask_token_id)
        masked_positions = torch.where(still_masked)[0]

        if masked_positions.numel() == 0:
            break

        # Get top-1 (best candidate) for each still-masked position
        best_probs = top_probs[masked_positions, 0]   # [num_still_masked]
        best_ids = top_token_ids[masked_positions, 0] # [num_still_masked]

        # How many to fill this step
        fill_count = min(top_k, masked_positions.numel())

        # Pick the top-K most confident positions
        top_indices = torch.topk(best_probs, k=fill_count, dim=-1).indices

        # Fill all K positions at once
        for idx in top_indices:
            pos = masked_positions[idx].item()
            token = best_ids[idx].item()
            x[0, pos] = token

    output_text = tokenizer.decode(x.cpu().tolist()[0])
    print(output_text)
    return output_text

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lm = torch.load(TS_MODEL_PATH, weights_only=False, map_location=torch.device(device))   # MAX_SEQ_LENGTH = 256
my_tokenizer = get_tokenizer()
# prompt = 'Once upon a time, there was a little girl'
prompt = "<|endoftext|>"
seq_length = 192 # MAX_SEQ_LENGTH 256
top_k = 1

output = inference_simple(
    model=lm,
    tokenizer=my_tokenizer,
    device=device,
    prompt=prompt,
    seq_length=seq_length,
    temperature=1.00,
    top_k=top_k
)