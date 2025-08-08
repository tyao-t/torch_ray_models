from transformer import TransformerBlock
import torch
import torch.nn as nn
import math

GPT_CONFIG_124M = {
    "vocab_size": 50304,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "bias": False
}

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.vocab_size is not None
        assert cfg.block_size is not None
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *( TransformerBlock(cfg) for _ in range(cfg["n_layers"]) ) 
        )
        
        self.ln_final = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * cfg["n_layer"]))

        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def forward(self, in_idx):
        batch_size, num_tokens = in_idx.shape
        assert num_tokens <= self.cfg["context_length"], \
            f"Unable to forward {num_tokens} tokens, max context_length is {self.cfg["context_length"]}"

        tok_embeds = self.tok_emb(in_idx) # (batch_size, num_tokens, emb_dim)
        # (num_tokens, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(num_tokens, device=in_idx.device))
        
        x = tok_embeds + pos_embeds  # (batch_size, num_tokens, emb_dim)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.ln_final(x)

        logits = self.out_head(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # elif isinstance(module, nn.LayerNorm): 
        #     if module.weight is not None: torch.nn.init.ones_(module.weight)
        #     if module.bias is not None: torch.nn.init.zeros_(module.bias)
        #  The above last elif has been done by default for nn.LayerNorm

def generate_text_simple(model, idx, max_new_tokens, context_length):
    batch, num_tokens = idx.shape
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        
        with torch.no_grad():
            logits = model(idx_cond) # (batch, min(context_length, num_tokens), vocab_size)

        logits = logits[:, -1, :]  # (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def generate(model, idx, max_new_tokens, context_length, temperature=0.0, top_k=None, eos_id=None):
    def softmax_with_temperature(logits, temperature):
        scaled_logits = logits / temperature
        return torch.softmax(scaled_logits, dim=-1)

    batch, num_tokens = idx.shape

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond) # (batch, min(context_length, num_tokens), vocab_size)
        next_token_logits = logits[:, -1, :]

        if not top_k:
            idx_next = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            top_logits, top_pos = torch.topk(next_token_logits, top_k)
            # min_val = top_logits[:, -1]
            # logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
            
            # A relatively better implementation:
            logits = torch.full_like(
                next_token_logits, -torch.inf
            )   
            logits[top_pos] = next_token_logits[top_pos]

            if temperature > 0.0:
                probs = softmax_with_temperature(logits, temperature)
            
            idx_next = torch.multinomial(probs, num_samples=1)            

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_length = model.cfg["context_length"] # model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_length=context_length
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer).replace("\n", " ") 
        print(decoded_text)
    model.train()