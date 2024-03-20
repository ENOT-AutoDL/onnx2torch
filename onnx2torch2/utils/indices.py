import torch

__all__ = [
    'upcast_indices',
]

_INT_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def upcast_indices(indices: torch.Tensor) -> torch.Tensor:
    """
    Upcasts indices tensor to torch.int64 (long) dtype.

    indices : torch.Tensor
        Indices for upcasting to torch.int64.

    Returns
    -------
    torch.Tensor
        Upcasted to torch.int64 tensor.

    """
    if not any(indices.dtype == dtype for dtype in _INT_DTYPES):
        raise ValueError(f'Expected types of indices: {_INT_DTYPES}, got {indices.dtype} instead')
    return indices.type(dtype=torch.int64)
