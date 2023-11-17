import torch
import torch.nn as nn

from icecream import ic


class MyModule(nn.Module):

    def __init__(self):
        super().__init__()

        self.register_forward_hook(self._forward_hook)
        self.register_full_backward_hook(self._backward_hook)

        self.forward_features = {}
        self.backward_features = {}

    def _forward_hook(self, module, input, output):
        self.forward_features['input'] = input
        self.forward_features['output'] = output
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.backward_features['grad_input'] = grad_input
        self.backward_features['grad_output'] = grad_output

    def forward(self, x, y):

        return x + y, x - y
    

module = MyModule()
input1 = torch.tensor([1], dtype=torch.float32, requires_grad=True)
input2 = torch.tensor([2], dtype=torch.float32, requires_grad=True)
pred1, pred2 = module(input1, input2)
loss = torch.square(pred1 + pred2)
loss.backward()
ic(input1, input2, pred1, pred2)
ic(module.forward_features, module.backward_features)