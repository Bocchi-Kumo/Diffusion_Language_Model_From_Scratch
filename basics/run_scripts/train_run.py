from basics.model import diffusion_lm
from basics.nn_utils import gradient_clipping, forward_diffusion_absorb, denoising_score_entropy
from basics.data import data_loading
from basics.optimizer import AdamW
from basics.run_scripts.filepath import *
import torch
from basics.helper_function import get_tokenizer
import numpy as np
from tqdm import tqdm

def validate(model, dataset, loss_fn, batch_size, context_length, num_timesteps, mask_token_id, device, val_steps=10):
    """Run validation for a fixed number of steps."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(val_steps):
            x_0 = data_loading(dataset, batch_size, context_length, device)
            t = torch.randint(low=0, high=num_timesteps, size=(batch_size, 1), device=device)
            t = t.float() / num_timesteps
            x_t, sigma, bar_sigma, a= forward_diffusion_absorb(x_0, t, mask_token_id)
            logits = model(x_t, t)
            loss = loss_fn(logits, x_0, x_t, mask_token_id, sigma)
            total_loss += loss.item()
    model.train()
    return total_loss / val_steps

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab_size = 10000
batch_size = 64
context_length = 256
d_model = 512
num_layers = 4
num_heads = 16
d_ff = 1344
rope_theta = 10000
num_timesteps = 256

training_loop = 100000

my_tokenizer = get_tokenizer()
mask_token_id = my_tokenizer.encode("<|mask|>")[0]

# Tiny Story
training_dataset = np.memmap(filename=TS_TRAIN_IDS_PATH, dtype=np.uint16)
validation_dataset = np.memmap(filename=TS_VALID_IDS_PATH, dtype=np.uint16)

lm = diffusion_lm(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    rope_theta=rope_theta
).to(device)

optimizer = AdamW(lm.parameters(), lr=5e-5)
loss_fn = denoising_score_entropy

# --- Training loop ---
for i in tqdm(range(training_loop)):
    x_0 = data_loading(training_dataset, batch_size, context_length, device)
    t = torch.randint(low=0, high=num_timesteps, size=(batch_size, 1), device=device)
    t = t.float() / num_timesteps
    x_t, sigma, bar_sigma, a = forward_diffusion_absorb(x_0, t, mask_token_id)
    logits = lm(x_t, t) # Out put from model is log_logits, s = exp(log_ligts)
    loss = loss_fn(logits, x_0, x_t, mask_token_id, sigma)
    optimizer.zero_grad()
    loss.backward()
    gradient_clipping(lm.parameters(), max_l2_norm=1.0)
    optimizer.step()

    if (i + 1) % 100 == 0:
        val_loss = validate(lm, validation_dataset, loss_fn, batch_size, context_length,
                           num_timesteps, mask_token_id, device, val_steps=50)
        print(f"Step {i+1} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss:.5f}")

torch.save(lm, TS_MODEL_PATH)
print("Training done")
lm = torch.load(TS_MODEL_PATH, weights_only=False, map_location=torch.device(device))
print("Loaded model")
val_loss = validate(lm, validation_dataset, loss_fn, batch_size, context_length,
                    num_timesteps, mask_token_id, device, val_steps=100)
print(val_loss)