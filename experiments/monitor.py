import functools
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

def detect_nans(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        if isinstance(result, torch.Tensor):
            if torch.isnan(result).any():
                raise ValueError(f"NaN detected in {func.__name__}")
            if torch.isinf(result).any():
                raise ValueError(f"Inf detected in {func.__name__}")
                
        if isinstance(result, CausalLMOutputWithPast):
            item = result.logits
            if torch.isnan(item).any():
                raise ValueError(f"NaN detected in {func.__name__}")
            if torch.isinf(item).any():
                raise ValueError(f"Inf detected in {func.__name__}")
        
        return result
    return wrapper
