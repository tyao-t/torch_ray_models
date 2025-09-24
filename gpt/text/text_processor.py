import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, window_size, stride, *, encode=True):
        self.input_ids, self.target_ids = [], []

        token_ids = text
        if encode: text_to_token_ids(token_ids, tokenizer, create_batch_dim=False)
        assert len(token_ids) > window_size # > is due to target_chunk = token_ids[i+1:i+window_size+1]

        for i in range(0, len(token_ids) - window_size, stride):
            input_chunk = token_ids[i:i+window_size]
            target_chunk = token_ids[i+1:i+window_size+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
        

def create_dataloader(text, batch_size=4, window_size=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0, *, encode=True):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDataset(text, tokenizer, window_size, stride, encode=encode)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, \
        drop_last=drop_last, num_workers=num_workers, \
        pin_memory=True # Optimized, new version
    )

    return dataloader


def text_to_token_ids(text, tokenizer, *, create_batch_dim=True):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded)
    if create_batch_dim:
        encoded_tensor = encoded_tensor.unsqueeze(0)
    return encoded_tensor # (1, batch_size)

def token_ids_to_text(token_ids, tokenizer, *, remove_batch_dim=True):
    flat = token_ids.squeeze(0) if remove_batch_dim else token_ids  # (1, batch_size)
    return tokenizer.decode(flat.tolist()) # (batch_size, )

def one_hot(idx):
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    onehot = torch.nn.functional.one_hot(idx)
    torch.manual_seed(123)
    num_idx = max(idx)+1 # or it can be vocab_size
    linear = torch.nn.Linear(num_idx, out_dim, bias=False)
    linear.weight = torch.nn.Parameter(embedding.weight.T)
    linear.weight
    linear(onehot.float()) == embedding(idx)
    
if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        dataloader = create_dataloader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
        data_iter = iter(dataloader)
        first_batch = next(data_iter)
        second_batch = next(data_iter)
        print(second_batch)

    vocab_size, output_dim = 6, 3
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(first_batch)
    print(token_embeddings.shape)
    # torch.Size([8, 4, 256])

    context_length = 1024
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)
    # torch.Size([4, 256])

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)
    # torch.Size([8, 4, 256])