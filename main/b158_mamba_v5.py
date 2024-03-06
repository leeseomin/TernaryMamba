import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from mamba_ssm import Mamba
import matplotlib.pyplot as plt
from IPython import display

# hyperparams
epochs = 100
lr = 1e-3
batch_size = 64
block_size = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
max_iters = 10000
print_iters = 100
eval_iters = 10
eval_interval = 300
n_embed = 384
n_heads = 6
n_layers = 6
dropout = 0.2
num_groups = 4

# Load and process data
with open("input.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda xx: [stoi[x] for x in xx]
decode = lambda xx: ''.join([itos[x] for x in xx])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

# Mamba Block
class MambaBlock(nn.Module):
    def __init__(self, d_model, expansion_factor, SSM_dim, num_groups=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.linear1 = nn.Linear(d_model, expansion_factor*d_model)
        self.linear2 = nn.Linear(expansion_factor*d_model, d_model)
        self.num_groups = num_groups
        self.eps = 1e-5
        
        self.activation = nn.Hardtanh(-1, 1)
        self.SSM = Mamba(d_model=d_model, d_state=SSM_dim, d_conv=4, expand=expansion_factor).to(device)

    def ste_quantize(self, x):
        # Apply the sign function for ternarization
        quantized_x = torch.sign(x)
        # Use STE: during backward pass, we bypass the quantization
        quantized_x = (quantized_x - x).detach() + x
        return quantized_x

    def quantize_weights_groupwise(self, weights):
        # Divide weights into groups
        group_size = weights.shape[1] // self.num_groups
        quantized_weights = torch.zeros_like(weights)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = weights[:, start_idx:end_idx]

            # Quantize weights
            quantized_weights[:, start_idx:end_idx] = self.ste_quantize(weight_group)

        return quantized_weights

    def quantize_activations_groupwise(self, x, b=8):
        Q_b = 2 ** (b - 1)

        # Divide activations into groups
        group_size = x.shape[1] // self.num_groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[:, start_idx:end_idx]

            # Quantize each group
            gamma_g = activation_group.abs().max()
            quantized_x[:, start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x

    def forward(self, x):
        # Quantize weights (group-wise) using STE
        quantized_weights1 = self.quantize_weights_groupwise(self.linear1.weight)
        quantized_weights2 = self.quantize_weights_groupwise(self.linear2.weight)

        gated = self.activation(F.linear(self.ln1(x), quantized_weights1, self.linear1.bias))
        x = x + F.linear(gated, quantized_weights2, self.linear2.bias)

        # Quantize activations group-wise
        quantized_x = self.quantize_activations_groupwise(x)
        
        x = quantized_x + self.SSM(self.ln2(quantized_x))
        return x

# Language Model
class MambaLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, expansion_factor, SSM_dim, num_groups=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.num_groups = num_groups
        
        self.layers = nn.ModuleList([MambaBlock(d_model, expansion_factor, SSM_dim, num_groups) 
                                     for _ in range(num_layers)])
        
        self.ln_out = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_out(x)
        logits = self.proj(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
# Training and evaluation
model = MambaLM(vocab_size, n_embed, n_layers, expansion_factor=4, SSM_dim=16, num_groups=num_groups).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        data = train_data if split == 'train' else val_data
        losses[split] = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[split][k] = loss.item()
    model.train()
    return losses

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Initialize the figure and axes for plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
train_losses = []
val_losses = []

for iter in tqdm(range(max_iters)):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}  |  train loss {losses['train'].mean():.4f}  |  val loss {losses['val'].mean():.4f}")
        
        train_losses.append(losses['train'].mean())
        val_losses.append(losses['val'].mean())

        ax.clear()
        ax.plot(train_losses, label='Train Loss')
        ax.plot(val_losses, label='Val Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
    if iter % print_iters == 0:
        model.eval()
        sample_idx = model.generate(torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=200)
        print(decode(sample_idx[0].tolist()))
        model.train()

# Generate text
model.eval()
sample_idx = model.generate(torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=1000)
print(decode(sample_idx[0].tolist()))


