import torch
import torch.nn.functional as F


def masked_mean(tensor, mask, dim=None, keepdim=False):
    """
    Compute mean of tensor with a boolean mask.
    
    Args:
        tensor: Input tensor
        mask: Boolean mask (True for valid elements, False for masked out)
        dim: Dimension(s) to reduce. If None, reduces all dimensions.
        keepdim: Whether to keep the reduced dimensions
    
    Returns:
        Masked mean
    """
    masked_tensor = torch.where(mask, tensor, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    sum_valid = masked_tensor.sum(dim=dim, keepdim=keepdim)
    count_valid = mask.sum(dim=dim, keepdim=keepdim)
    
    # Avoid division by zero
    count_valid = count_valid.clamp(min=1)
    
    return sum_valid / count_valid


def masked_var(tensor, mask, dim=None, keepdim=False, unbiased=True):
    """
    Compute variance of tensor with a boolean mask.
    
    Args:
        tensor: Input tensor
        mask: Boolean mask (True for valid elements, False for masked out)
        dim: Dimension(s) to reduce. If None, reduces all dimensions.
        keepdim: Whether to keep the reduced dimensions
        unbiased: If True, use Bessel's correction (divide by N-1 instead of N)
    
    Returns:
        Masked variance
    """
    # Compute masked mean
    mean = masked_mean(tensor, mask, dim=dim, keepdim=True)
    
    # Compute squared deviations
    squared_diff = (tensor - mean) ** 2
    masked_squared_diff = torch.where(mask, squared_diff, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    
    sum_squared_diff = masked_squared_diff.sum(dim=dim, keepdim=keepdim)
    count_valid = mask.sum(dim=dim, keepdim=keepdim)
    
    # Avoid division by zero
    if unbiased:
        count_valid = (count_valid - 1).clamp(min=1)
    else:
        count_valid = count_valid.clamp(min=1)
    
    return sum_squared_diff / count_valid

def masked_softmax(tensor, mask, dim=-1):
    """
    Compute softmax with a boolean mask.
    
    Args:
        tensor: Input tensor (logits)
        mask: Boolean mask (True for valid elements, False for masked out)
        dim: Dimension along which to apply softmax
    
    Returns:
        Masked softmax probabilities (masked positions will be near-zero)
    """
    # Set masked positions to negative infinity
    masked_tensor = tensor.masked_fill(~mask, float('-inf'))

    # Apply softmax
    return F.softmax(masked_tensor, dim=dim) * mask

def masked_log_softmax(tensor, mask, dim=-1, mask_value=-1e9):
    """
    Compute log_softmax with a boolean mask.
    
    Args:
        tensor: Input tensor (logits)
        mask: Boolean mask (True for valid elements, False for masked out)
        dim: Dimension along which to apply log_softmax
    
    Returns:
        Masked log_softmax (masked positions will be -inf)
    """
    masked_tensor = tensor.masked_fill(~mask, float('-inf'))

    log_probs = F.log_softmax(masked_tensor, dim=dim)

    log_probs = log_probs.masked_fill(~mask, mask_value)
    
    return log_probs


# Taken from https://arxiv.org/pdf/2403.17031 then modified to add masking
def masked_whiten(values, mask, shift_mean=True):
    mean, var = masked_mean(values, mask), masked_var(values, mask, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened * mask
