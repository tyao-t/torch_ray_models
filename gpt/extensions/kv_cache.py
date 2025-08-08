import torch
from typing import Optional, Tuple

class TinyKvCache:
    def update_and_fetch(
        self, 
        key: torch.Tensor, # (batch_size, num_heads, num_tokens, head_dim)
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]: # (keys, values, start_offset)
        raise NotImplementedError("Derived classes must implement update_and_fetch")


class TinyKvFullCache(TinyKvCache):
    """Grows indefinitely by concatenation"""
    def __init__(self):
        super().__init__()
        self.key_values = None # (keys, values)
        self.offset = 0
    
    def update_and_fetch(
        self, 
        key: torch.Tensor, 
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.key_values is None:
            assert self.offset == 0
            self.key_values = (key, value)
            batch_size, num_heads, num_tokens, head_dim = key.shape
            self.offset = num_tokens
            return key, value, 0
        else:
            batch_size, num_heads, num_tokens, head_dim = key.shape
            assert key.shape == value.shape
            prev_keys, prev_values = self.key_values
            
            # Verify shapes match
            assert prev_keys.shape == (batch_size, num_heads, self.offset, head_dim)
            assert prev_values.shape == (batch_size, num_heads, self.offset, head_dim)
            
            new_keys = torch.cat([prev_keys, key], dim=2)
            new_values = torch.cat([prev_values, value], dim=2)
            
            self.key_values = (new_keys, new_values)
            start_offset = self.offset
            self.offset += num_tokens
            
            return new_keys, new_values, start_offset

class TinyKvRotatingCache(TinyKvCache):
    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.key_values = None # (keys, values)
        self.head = 0 
        self.head_offset = 0
    
    def update_and_fetch(
        self, key: torch.Tensor, value: torch.Tensor, offset: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_heads, num_tokens, head_dim = key.shape
        
        if self.key_values is None:
            assert offset == 0, "First call must have offset=0"
            assert num_tokens <= self.max_seq_len, f"Input length {num_tokens} exceeds max {self.max_seq_len}"
            
            keys = torch.zeros((batch_size, num_heads, self.max_seq_len, head_dim), 
                              dtype=key.dtype, device=key.device)
            values = torch.zeros_like(keys)
            
            keys[:, :, :num_tokens, :] = key
            values[:, :, :num_tokens, :] = value
            
            self.key_values = (keys, values)
            self.head = num_tokens
            self.head_offset = num_tokens
            
            return keys[:, :, :num_tokens, :], values[:, :, :num_tokens, :]
        
        else:
            assert key.shape == value.shape, "Key/value shape mismatch"
            assert offset == self.head_offset, f"Offset mismatch: expected {self.head_offset}, got {offset}"
            assert num_tokens <= self.max_seq_len, f"Input length {num_tokens} exceeds max {self.max_seq_len}"
            
            keys, values = self.key_values
            
            if self.head + num_tokens <= self.max_seq_len:
                keys[:, :, self.head:self.head+num_tokens] = key
                values[:, :, self.head:self.head+num_tokens] = value
                self.head += num_tokens
                self.head_offset += num_tokens
            
            else:
                fill_size = self.max_seq_len - self.head
                keys[:, :, self.head:] = key[:, :, :fill_size]
                values[:, :, self.head:] = value[:, :, :fill_size]
                
                remaining_size = num_tokens - fill_size
                keys[:, :, :remaining_size] = key[:, :, fill_size:]
                values[:, :, :remaining_size] = value[:, :, fill_size:]
                
                self.head = remaining_size
                self.head_offset += num_tokens
            
            self.key_values = (keys, values)
            
            if self.head_offset < self.max_seq_len:
                return keys[:, :, :self.head_offset], values[:, :, :self.head_offset]
            else:
                before_keys = keys[:, :, self.head:]
                before_values = values[:, :, self.head:]
                after_keys = keys[:, :, :self.head]
                after_values = values[:, :, :self.head]
                
                keys_contiguous = torch.cat([after_keys, before_keys], dim=2)
                values_contiguous = torch.cat([after_values, before_values], dim=2)
                
                return keys_contiguous, values_contiguous