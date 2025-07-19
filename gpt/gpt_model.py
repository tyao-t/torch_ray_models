from transformer import TransformerBlock
import torch
import torch.nn as nn
import tiktoken

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # (batch_size, num_tokens, emb_size)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx.shape == (batch, n_tokens)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # (batch, min(context_size, n_tokens), vocab_size)
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    # out = model(batch)
    # print("Input batch:\n", batch)
    # print("\nOutput shape:", out.shape)
    # """
    # Input batch:
    # tensor([[6109, 3626, 6100, 345],
    #         [6109, 1110, 6622, 257]])
    # Output shape: torch.Size([2, 4, 50257])"""

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context) # (n_tokens)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # (1, n_tokens)
    # encoded: [15496, 11, 314, 716]
    # encoded_tensor.shape: torch.Size([1, 4])

    model.eval() # disable dropout

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor, 
        max_new_tokens=6, 
        context_size=GPT_CONFIG_124M["context_length"]
    )
    # out: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267]])

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
