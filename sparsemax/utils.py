import torch


def flatten_all_but_nth_dim(ctx, x: torch.Tensor):
    """
    Flattens tensor in all but 1 chosen dimension.
    Saves necessary context for backward pass and unflattening.
    """

    # transpose batch and nth dim
    x = x.transpose(0, ctx.dim)

    # Get and save original size in context for backward pass
    original_size = x.size()
    ctx.original_size = original_size

    # Flatten all dimensions except nth dim
    x = x.reshape(x.size(0), -1)

    # Transpose flattened dimensions to 0th dim, nth dim to last dim
    return ctx, x.transpose(0, -1)


def unflatten_all_but_nth_dim(ctx, x: torch.Tensor):
    """
    Unflattens tensor using necessary context
    """
    # Tranpose flattened dim to last dim, nth dim to 0th dim
    x = x.transpose(0, 1)

    # Reshape to original size
    x = x.reshape(ctx.original_size)

    # Swap batch dim and nth dim
    return ctx, x.transpose(0, ctx.dim)
