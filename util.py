def get_default_strides(shape: tuple|list) -> list:
    suffix = 1
    strides = []
    for i in reversed(shape):
        strides.insert(0,suffix)
        suffix *= i
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