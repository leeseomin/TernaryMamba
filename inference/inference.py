import torch
from mamba_ssm import Mamba
from safetensors.torch import load_file
import argparse

# Hyperparameters
block_size = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embed = 384
n_heads = 6
n_layers = 6
dropout = 0.2

# Load the character mappings
with open("chars.txt", "r") as f:
    chars = f.read().split(',')

vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda xx: [stoi[x] for x in xx]
decode = lambda xx: ''.join([itos[x] for x in xx])

# Mamba Block
class MambaBlock(nn.Module):
    def __init__(self, d_model, expansion_factor, SSM_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.linear1 = nn.Linear(d_model, expansion_factor*d_model)
        self.linear2 = nn.Linear(expansion_factor*d_model, d_model)
        
        self.activation = nn.Hardtanh(-1, 1)
        self.SSM = Mamba(d_model=d_model, d_state=SSM_dim, d_conv=4, expand=expansion_factor).to(device)

    def forward(self, x):
        gated = self.activation(self.linear1(self.ln1(x)))
        x = x + self.linear2(gated)
        x = x + self.SSM(self.ln2(x))
        return x

# Language Model
class MambaLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, expansion_factor, SSM_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([MambaBlock(d_model, expansion_factor, SSM_dim) 
                                     for _ in range(num_layers)])
        
        self.ln_out = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_out(x)
        logits = self.proj(x)
        return logits
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.safetensors', help='Path to the model file')
    parser.add_argument('--text', type=str, default='Hello', help='Input text for generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    args = parser.parse_args()

    model = MambaLM(vocab_size, n_embed, n_layers, expansion_factor=4, SSM_dim=16).to(device)
    model.load_state_dict(load_file(args.model_path))
    model.eval()

    input_ids = torch.tensor(encode(args.text), dtype=torch.long).unsqueeze(0).to(device)
    output_ids = model.generate(input_ids, max_new_tokens=args.max_length)
    output_text = decode(output_ids[0].tolist())

    print(output_text)

if __name__ == '__main__':
    main()

    