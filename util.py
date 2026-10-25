import numpy as np

def get_default_strides(shape: tuple|list) -> list:
    from tensor import sint
    suffix = 1
    strides = []
    for i in reversed(shape):
        strides.insert(0,suffix)
        suffix = i * suffix if isinstance(i, sint) else suffix * i
    return strides

def canonicalize_index(key, shape):
    if not isinstance(key, tuple):
        key = (key,) 
    if len(shape) > len(key):
        key += tuple(slice(None,None,None) for _ in range(len(shape) - len(key)))
    assert len(key) == len(shape)
    return key

def get_new_shape_from_index(key, shape, strides, offset):
    assert len(key) == len(shape) and len(key) == len(strides)
    offset = 0 if offset is None else offset
    new_strides = list(strides)
    new_shape = list(shape)
    for i,idx in enumerate(reversed(key)):
        i = len(key) - i - 1
        if isinstance(idx, slice):
            start, end, step = idx.indices(shape[i])
            assert start < end, "invalid index"
            offset += start*new_strides[i]
            new_strides[i] = new_strides[i] * step
            new_shape[i] = (end-start)//step 
        elif isinstance(idx, int):
            assert idx < new_shape[i], "index out of bound"
            offset += idx*strides[i]
            new_strides[i:] = new_strides[i+1:] 
            new_shape[i:] = new_shape[i+1:]
    return new_shape, new_strides, offset

def is_tensor_dyshape(tensor):
    from tensor import sint
    return any(isinstance(x, sint) for x in tensor._shape)

def align_tensor_shape(tensor, other):
    from tensor import sint
    is_a_dyshape = is_tensor_dyshape(tensor)
    is_b_dyshape = is_tensor_dyshape(other)
    assert not (is_a_dyshape and is_b_dyshape), "can not align two dyshape tensor"
    zipped = zip(tensor._shape, other._shape) if is_a_dyshape else zip(other._shape, tensor._shape)
    new_shape = tuple(max(s1, s2) if not isinstance(s1, sint) else s1 for s1, s2 in zipped)
    return tensor.expand(*new_shape), other.expand(*new_shape)
        
def get_real_shape(tensor):
    from tensor import sint
    new_shape = tensor._shape
    if new_shape is not None:
        for i,x in enumerate(tensor._shape):
            if isinstance(x, sint):
                if isinstance(x.data, int):
                    new_shape[i] = x.data
    return new_shape

def get_real_strides(tensor):
    from tensor import sint
    new_strides = tensor._strides
    if new_strides is not None:
        for i,x in enumerate(tensor._strides):
            if isinstance(x, sint):
                if isinstance(x.data, int):
                    new_strides[i] = x.data
    return new_strides

def get_real_offset(tensor):
    from tensor import sint
    new_offset = tensor._offset
    if new_offset is not None and isinstance(tensor._offset, sint):
        if isinstance(tensor._offset.data, int):
            new_offset = tensor._offset.data
    return new_offset

def dyint_in_tensor(tensor):
    # return all dynamic int in tensor shape, strides, offset
    from tensor import sint
    dyints = []
    if tensor._shape is not None:
        for i,x in enumerate(tensor._shape):
            if isinstance(x, sint):
                dyints.append(x)
    if tensor._strides is not None:
        for i,x in enumerate(tensor._strides):
            if isinstance(x, sint):
                dyints.append(x)
    if tensor._offset is not None and isinstance(tensor._offset, sint):
        dyints.append(tensor._offset)
    return dyints

def bind_tensor_dyshape(tensor, input):
    from tensor import sint
    if tensor._shape is not None:
        for i,x in enumerate(tensor._shape):
            if isinstance(x, sint):
                assert input.shape[i] >= x.min and input.shape[i] <= x.max
                x.bind(input.shape[i])
    if tensor._strides is not None:
        for i,x in enumerate(tensor._strides):
            if isinstance(x, sint):
                assert input.strides[i] >= x.min and input.strides[i] <= x.max
                x.bind(input.strides[i])
    if tensor._offset is not None and isinstance(tensor._offset, sint):
        assert input.offset >= tensor._offset.min and input.offset <= tensor._offset.max
        tensor._offset.bind(input.offset)