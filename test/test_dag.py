import numpy as np
import sys

sys.path.append("..")
from tensor import Tensor

def test_dag():
    a = Tensor(1)
    b = Tensor(2)
    c = Tensor(3)
    d = a + b
    e = b + c
    f = d + e
    print(f.lower())


if __name__ == "__main__":
    test_dag()