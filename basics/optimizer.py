from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=[0.9, 0.999], eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data) # Initial value of the first moment vector; same shape as θ
                if "v" not in state:
                    state["v"] = torch.zeros_like(p.data) # Initial value of the second moment vector; same shape as θ
                t = state.get("t", 0) + 1 # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                # state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                # state['v'] = beta2 * state['v'] + (1 - beta2) * (grad ** 2)
                state['m'].mul_(beta1).add_(grad, alpha=1 - beta1)
                state['v'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)                
                # m_hat = state['m'] / (1 - beta1 ** t)
                # v_hat = state['v'] / (1 - beta2 ** t)
                p.data -= lr * (state['m'] / (1 - beta1 ** t)) / (torch.sqrt(state['v'] / (1 - beta2 ** t)) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t
        return loss
    
def lr_cosine_schedule(t, lr_max, lr_min, T_w, T_c):
    '''
    Args:
        t: current iteration
        lr_max: maximun learning rate
        lr_min: minimum(final) learning rate
        T_w: number of warm-up iterations
        T_c: number of cosine annealing iterations
    output:
        the current learning rate
    '''
    if (t < 0):
        raise ValueError("Iteration t must be >= 0, got t={}".format(t))
    # Warm-up
    if(t < T_w):
        lr_t = (t/T_w) * lr_max
    # Cosine annealing
    elif(T_w <= t and t <= T_c):
        lr_t = lr_min + 0.5 * (1 + math.cos(((t-T_w) / (T_c-T_w) * math.pi)))*(lr_max - lr_min)
    # Post-annealing
    else:
        lr_t = lr_min
    return lr_t