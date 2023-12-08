import torch
import torch.nn.functional as F
from icecream import ic

# batch_size = 2
# num_classes = 3
# input_size = 4

# labels = torch.randint(0, num_classes, (batch_size,))
# one_hot_labels = F.one_hot(labels, num_classes=num_classes)
# ic(one_hot_labels)

# x = torch.randint(0, num_classes, (batch_size, input_size), dtype=torch.float32)
# w = torch.randn(input_size, num_classes, requires_grad=True)
# logits = x @ w
# ic(x, w, logits)

# masked_logits = torch.sum(logits * one_hot_labels)
# ic(masked_logits)

# masked_logits.backward(retain_graph=True)
# ic(w.grad)

# masked_logits.backward(gradient=masked_logits, retain_graph=True)
# ic(w.grad)

number = 1
if not number:
    print('number')