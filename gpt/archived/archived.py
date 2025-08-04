# gelu_exact = nn.GELU(approximate='none')  
# gelu_approx = nn.GELU(approximate='tanh') 

# def _init_weights(self, module):
#     if isinstance(module, nn.Linear):
#         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#         if module.bias is not None:
#             torch.nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.Embedding):
#         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    # elif isinstance(module, nn.LayerNorm): # this has been done by default for nn.LayerNorm
    #     if module.weight is not None: torch.nn.init.ones_(module.weight)
    #     if module.bias is not None: torch.nn.init.zeros_(module.bias)


# class GELU(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(
#             torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
#             (x + 0.044715 * torch.pow(x, 3))
#         ))

# class LayerNorm(nn.Module):
#     def __init__(self, emb_dim):
#         super().__init__()
#         self.eps = 1e-5
#         self.scale = nn.Parameter(torch.ones(emb_dim))
#         self.shift = nn.Parameter(torch.zeros(emb_dim))

#     def forward(self, x):
#         mean = x.mean(dim=-1, keepdim=True)
#         var = x.var(dim=-1, keepdim=True, unbiased=False)
#         norm_x = (x - mean) / torch.sqrt(var + self.eps)
#         return self.scale * norm_x + self.shift



# class TransformerBlockBootstrap(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.att = MultiHeadAttention(
#             d_in=cfg["emb_dim"],
#             d_out=cfg["emb_dim"],
#             context_length=cfg["context_length"],
#             num_heads=cfg["n_heads"],
#             dropout=cfg["drop_rate"],
#             bias=cfg["bias"]
#         )
#         self.ff = FeedForward(cfg)
#         self.norm1 = LayerNorm(cfg["emb_dim"], bias=cfg["bias"])
#         self.norm2 = LayerNorm(cfg["emb_dim"], bias=cfg["bias"])
#         self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
#     def forward(self, x):
#         shortcut = x
#         x = self.norm1(x)
#         x = self.att(x)
#         # x = self.drop_shortcut(x)
#         x = x + shortcut
        
#         shortcut = x
#         x = self.norm2(x)
#         x = self.ff(x)
#         # x = self.drop_shortcut(x)
#         x = x + shortcut
#         return x


# if __name__ == "__main__":
#     torch.manual_seed(123)
#     model = GPTModel(GPT_CONFIG_124M)

#     # out = model(batch)
#     # print("Input batch:\n", batch)
#     # print("\nOutput shape:", out.shape)
#     # """
#     # Input batch:
#     # tensor([[6109, 3626, 6100, 345],
#     #         [6109, 1110, 6622, 257]])
#     # Output shape: torch.Size([2, 4, 50257])"""

#     start_context = "Hello, I am"
#     encoded = tokenizer.encode(start_context) # (n_tokens)
#     encoded_tensor = torch.tensor(encoded).unsqueeze(0) # (1, n_tokens)
#     # encoded: [15496, 11, 314, 716]
#     # encoded_tensor.shape: torch.Size([1, 4])

#     model.eval() # disable dropout

#     out = generate_text_simple(
#         model=model,
#         idx=encoded_tensor, 
#         max_new_tokens=6, 
#         context_length=GPT_CONFIG_124M["context_length"]
#     )
#     # out: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267]])

#     decoded_text = tokenizer.decode(out.squeeze(0).tolist())

# def simple_demo_before_training():
#     inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
#                           [40, 1107, 588]])   #  "I really like"]

#     targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
#                             [1107,  588, 11311]]) #  " really like chocolate"]

#     with torch.no_grad():
#         logits = model(inputs)

#     probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
#     print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)

#     token_ids = torch.argmax(probas, dim=-1, keepdim=True)

#     print("Logits shape:", logits.shape)
#     print("Targets shape:", targets.shape)
#     # Logits shape: torch.Size([2, 3, 50257])
#     # Targets shape: torch.Size([2, 3])

#     logits_flat = logits.flatten(0, 1)
#     targets_flat = targets.flatten()

#     print("Flattened logits:", logits_flat.shape)
#     print("Flattened targets:", targets_flat.shape)
#     # Flattened logits: torch.Size([6, 50257])
#     # Flattened targets: torch.Size([6])

#     loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
#     perplexity = torch.exp(loss)

#     vocab = {
#         "closer": 0,
#         "every": 1,
#         "effort": 2,
#         "forward": 3,
#         "inches": 4,
#         "moves": 5,
#         "pizza": 6,
#         "toward": 7,
#         "you": 8,
#     }

#     inverse_vocab = {v: k for k, v in vocab.items()}
#     next_token_logits = torch.tensor(
#         [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
#     )
#     probas = torch.softmax(next_token_logits, dim=0)
#     # next_token_id = torch.argmax(probas).item()
#     next_token_id = torch.multinomial(probas, num_samples=1).item()
#     print(inverse_vocab[next_token_id])

#     top_k = 3
#     top_logits, top_pos = torch.topk(next_token_logits, top_k)
#     print("Top logits:", top_logits)
#     print("Top positions:", top_pos)

#     # Top logits: tensor([6.7500, 6.2800, 4.5100])
#     # Top positions: tensor([3, 7, 0])

#     new_logits = torch.where(
#         condition=next_token_logits < top_logits[-1],
#         input=torch.tensor(float("-inf")).to(next_token_logits.device), 
#         other=next_token_logits
#     )
#     print(new_logits)

#     # An alternative, slightly more efficient implementation of the previous code cell is the following:

#     new_logits = torch.full_like(
#         next_token_logits, -torch.inf
#     )   
#     new_logits[top_pos] = next_token_logits[top_pos]

 

#     # tensor([4.5100,   -inf,   -inf, 6.7500,   -inf,   -inf,   -inf, 6.2800,   -inf])

#     topk_probas = torch.softmax(new_logits, dim=0)
#     print(topk_probas)
#     # tensor([0.0615, 0.0000, 0.0000, 0.5775, 0.0000, 0.0000, 0.0000, 0.3610, 0.0000])

#     token_ids = generate(
#         model=model,
#         idx=text_to_token_ids("Every effort moves you", tokenizer),
#         max_new_tokens=15,
#         context_length=GPT_CONFIG_124M["context_length"],
#         top_k=25,
#         temperature=1.4
#     )

#     print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
#     #  Every effort moves you stand to work on surprise, a one of us had gone with random-

#     def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
#                        eval_freq, eval_iter, start_context, tokenizer):
#     train_losses, val_losses, track_tokens_seen = [], [], []
#     tokens_seen, global_step = 0, -1

#     for epoch in range(num_epochs):
#         model.train()
        
#         for input_batch, target_batch in train_loader:
#             optimizer.zero_grad()
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             loss.backward()
#             optimizer.step()
#             tokens_seen += input_batch.numel()
#             global_step += 1

#             if global_step % eval_freq == 0:
#                 train_loss, val_loss = evaluate_model(
#                     model, train_loader, val_loader, device, eval_iter)
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)
#                 track_tokens_seen.append(tokens_seen)
#                 print(f"Ep {epoch+1} (Step {global_step:06d}): "
#                       f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

#         generate_and_print_sample(
#             model, tokenizer, device, start_context
#         )

#     return train_losses, val_losses, track_tokens_seen

# def train_short():
#     file_path = "the-verdict.txt"
#     with open(file_path, "r", encoding="utf-8") as file:
#         tokenstext_data = file.read()

#     total_characters = len(text_data)
#     # total_tokens = len(tokenizer.encode(text_data))
#     print("Characters:", total_characters)
#     print("Tokens:", total_tokens)
#     # Characters: 20479, Tokens: 5145

#     tokenizer = tiktoken.get_encoding("gpt2")
#     tokens = text_to_token_ids(text_data, tokenizer, create_batch_dim=False)
#     train_ratio = 0.90
#     split_idx = int(train_ratio * len(tokens))
#     train_data = tokens[:tsplit_idxokens]
#     val_data = tokens[split_idx:]

#     torch.manual_seed(123)
#     train_loader = create_dataloader_v1(
#         train_data,
#         batch_size=2,
#         max_length=GPT_CONFIG_124M["context_length"],
#         stride=GPT_CONFIG_124M["context_length"],
#         drop_last=True,
#         shuffle=True,
#         num_workers=0,
#         encode = False
#     )

#     val_loader = create_dataloader_v1(
#         val_data,
#         batch_size=2,
#         max_length=GPT_CONFIG_124M["context_length"],
#         stride=GPT_CONFIG_124M["context_length"],
#         drop_last=False,
#         shuffle=False,
#         num_workers=0,
#         encode = False
#     )


#     train_tokens, val_tokens = 0, 0
#     for input_batch, target_batch in train_loader:
#         train_tokens += input_batch.numel()

#     for input_batch, target_batch in val_loader:
#         val_tokens += input_batch.numel()

#     print("Training tokens:", train_tokens)
#     print("Validation tokens:", val_tokens)
#     print("All tokens:", train_tokens + val_tokens)
#     # Training tokens: 4608
#     # Validation tokens: 512
#     # All tokens: 5120

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     with torch.no_grad():
#         train_loss = calc_loss_loader(train_loader, model, device)
#         val_loss = calc_loss_loader(val_loader, model, device)

#     print("Training loss:", train_loss)
#     print("Validation loss:", val_loss)

#     torch.manual_seed(123)
#     model = GPTModel(GPT_CONFIG_124M)
#     model.to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

#     num_epochs = 10
#     train_losses, val_losses, tokens_seen = train_model_simple(
#         model, train_loader, val_loader, optimizer, device,
#         num_epochs=num_epochs, eval_freq=5, eval_iter=5,
#         start_context="Every effort moves you", tokenizer=tokenizer
#     )


# if __name__ == "__main__":
#     train_short()