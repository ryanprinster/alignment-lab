import functools
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

def detect_nans(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        def check_value(value, name):
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any():
                    raise ValueError(f"NaN detected in {name}")
                if torch.isinf(value).any():
                    raise ValueError(f"Inf detected in {name}")
            elif isinstance(value, CausalLMOutputWithPast):
                if torch.isnan(value.logits).any():
                    raise ValueError(f"NaN detected in {name}")
                if torch.isinf(value.logits).any():
                    raise ValueError(f"Inf detected in {name}")
        
        if isinstance(result, (torch.Tensor, CausalLMOutputWithPast)):
            check_value(result, func.__name__)
        elif isinstance(result, (tuple, list)):
            for i, item in enumerate(result):
                check_value(item, f"{func.__name__}[{i}]")
        elif isinstance(result, dict):
            for key, value in result.items():
                check_value(value, f"{func.__name__}['{key}']")
        
        return result
    return wrapper
