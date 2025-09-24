base_tokenizer = tiktoken.get_encoding("gpt2")
custom_tokens = ["MyNewToken_1", "MyNewToken_2"]
custom_token_ids = {
    token: base_tokenizer.n_vocab + i for i, token in enumerate(custom_tokens)
}

extended_tokenizer = tiktoken.Encoding(
    name="gpt2_custom",
    pat_str=base_tokenizer._pat_str,
    mergeable_ranks=base_tokenizer._mergeable_ranks,
    special_tokens={**base_tokenizer._special_tokens, **custom_token_ids},
)

special_tokens_set = set(custom_tokens) | {"<|endoftext|>"}

token_ids = extended_tokenizer.encode(
    "Sample text with MyNewToken_1 and MyNewToken_2. <|endoftext|>",
    allowed_special=special_tokens_set
)

# gpt = GPTModel(...)
# gpt = None
# gpt.eval()

gpt.tok_emb # Embedding(50257, 768)

def extend_token_emb(gpt):
    num_tokens, emb_size = gpt.tok_emb.weight.shape
    new_num_tokens = num_tokens + 2
    new_embedding = torch.nn.Embedding(new_num_tokens, emb_size)
    new_embedding.weight.data[:num_tokens] = gpt.tok_emb.weight.data
    gpt.tok_emb = new_embedding

def extend_out_head(gpt, use_weight_tying=False):
    if use_weight_tying:
        gpt.out_head.weight = gpt.tok_emb.weight
        return

    original_out_features, original_in_features = gpt.out_head.weight.shape

    new_out_features = original_out_features + 2

    new_linear = torch.nn.Linear(original_in_features, new_out_features)

    with torch.no_grad():
        new_linear.weight[:original_out_features] = gpt.out_head.weight
        if gpt.out_head.bias is not None:
            new_linear.bias[:original_out_features] = gpt.out_head.bias

    gpt.out_head = new_linear
