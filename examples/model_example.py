import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from models.sharpen_focus import tensor_test, dataset_test


# tensor_test()
dataset_test()