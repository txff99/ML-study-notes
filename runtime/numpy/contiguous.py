from tensor import Tensor
import numpy as np

def contiguous_numpy_impl(src: Tensor) -> np.array:
    """
    this function return a flatten buffer
    """
    from util import get_default_strides
    shape = src.shape
    new_data = np.zeros(shape).flatten()
    assert src.is_realized, "src tensor need to be realized before making a contiguous tensor"
    old_data = src.data
    for ptr in range(np.prod(shape)):
        # map each element in old data to new data
        indices = np.zeros(len(shape))
        rem = ptr
        # compute index for new data
        for i,s in enumerate(get_default_strides(shape)):
            indices[i] = 0 if s == 0 else rem // s
            if s != 0:  rem %= s
        old_data_ptr = np.dot(indices, src.strides).astype(int)
        new_data[ptr] = old_data[old_data_ptr]
    return new_data
