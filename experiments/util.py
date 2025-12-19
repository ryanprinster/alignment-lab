import torch
import torch.nn.functional as F


def masked_mean(tensor, mask, dim=None, keepdim=False):
    masked_tensor = torch.where(
        mask, tensor, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
    )
    sum_valid = masked_tensor.sum(dim=dim, keepdim=keepdim)
    count_valid = mask.sum(dim=dim, keepdim=keepdim)

    # Avoid division by zero
    count_valid = count_valid.clamp(min=1)
    mean = sum_valid / count_valid
    return mean.to(tensor.dtype)


def masked_var(tensor, mask, dim=None, keepdim=False, unbiased=True):
    mean = masked_mean(tensor, mask, dim=dim, keepdim=True)

    # Compute squared deviations
    squared_diff = (tensor - mean) ** 2
    masked_squared_diff = torch.where(
        mask, squared_diff, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
    )

    sum_squared_diff = masked_squared_diff.sum(dim=dim, keepdim=keepdim)
    count_valid = mask.sum(dim=dim, keepdim=keepdim)

    # Avoid division by zero
    if unbiased:
        count_valid = (count_valid - 1).clamp(min=1)
    else:
        count_valid = count_valid.clamp(min=1)

    var = sum_squared_diff / count_valid
    return var.to(tensor.dtype)


def masked_softmax(tensor, mask, dim=-1, mask_value=-1e9):
    masked_tensor = tensor.masked_fill(~mask, float("-inf"))
    result = F.softmax(masked_tensor, dim=dim)
    result = result.masked_fill(~mask, mask_value)
    return result.to(tensor.dtype)


def masked_log_softmax(tensor, mask, dim=-1, mask_value=-1e9):
    masked_tensor = tensor.masked_fill(~mask, float("-inf"))
    log_probs = F.log_softmax(masked_tensor, dim=dim)
    log_probs = log_probs.masked_fill(~mask, mask_value)
    return log_probs.to(tensor.dtype)


# Taken from https://arxiv.org/pdf/2403.17031
def whiten(values, shift_mean=True):
    mean, var = torch.mean(values), torch.var(values)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened.to(values.dtype)


# Taken from https://arxiv.org/pdf/2403.17031 then modified to add masking
def masked_whiten(values, mask, shift_mean=True):
    mean, var = masked_mean(values, mask), masked_var(values, mask, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    result = whitened * mask
    return result.to(values.dtype)
