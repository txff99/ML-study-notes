def get_default_strides(shape: tuple) -> list:
    suffix = 1
    strides = []
    for i in reversed(shape):
        strides.insert(0,suffix)
        suffix *= i
    return strides