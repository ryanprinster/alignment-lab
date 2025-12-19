import functools
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

def detect_nans(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        def check_value(value, path):
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any() or torch.isinf(value).any():
                    raise ValueError(f"NaN/Inf detected in {path}")
            elif isinstance(value, CausalLMOutputWithPast):
                if torch.isnan(value.logits).any() or torch.isinf(value.logits).any():
                    raise ValueError(f"NaN/Inf detected in {path}")
            elif isinstance(value, (tuple, list)):
                for i, item in enumerate(value):
                    check_value(item, f"{path}[{i}]")
            elif isinstance(value, dict):
                for key, item in value.items():
                    check_value(item, f"{path}['{key}']")
        
        check_value(result, func.__name__)
        return result
    return wrapper