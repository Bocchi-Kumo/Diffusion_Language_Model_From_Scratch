import torch
from basics.model import diffusion_lm
from basics.optimizer import AdamW
from basics.nn_utils import forward_diffusion_absorb
from basics.test.test_score_entropy import denoising_score_entropy_naive


device = 'cpu'

vocab_size = 64
batch_size = 2
context_length = 32
d_model = 512
num_layers = 2
num_heads = 16
d_ff = 1344
rope_theta = 10000
num_timesteps = 1000
training_loop = 1000

mask_token_id = 0

x_0 = torch.randint(low=1, high=vocab_size, size=[batch_size, context_length]).to(device)
t = torch.randint(low=0, high=num_timesteps,size=(batch_size, 1)).to(device)
t = t.float() / num_timesteps
# x_t, sigma, a = forward_diffusion_absorb(x_0, t, mask_token_id, num_timesteps)

lm = diffusion_lm(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    rope_theta=rope_theta
).to(device)

optimizer = AdamW(lm.parameters(), lr=1e-4)
loss_fn = denoising_score_entropy_naive

# --- Training loop ---
for i in range(training_loop):
    x_t, sigma, a = forward_diffusion_absorb(x_0, t, mask_token_id)
    logits = lm(x_t, t)
    loss = loss_fn(logits, x_0, x_t, mask_token_id, sigma, a)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i, loss)