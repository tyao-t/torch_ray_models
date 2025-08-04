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