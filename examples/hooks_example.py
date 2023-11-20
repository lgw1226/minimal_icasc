import torch
import torch.nn as nn

from icecream import ic


class MyModule(nn.Module):

    def __init__(self):
        super().__init__()

        self.register_forward_hook(self._forward_hook)
        # from pytorch version 2, register_backward_hook is deprecated
        # register_full_backward_hook is basically same with register_backward_hook
        self.register_full_backward_hook(self._backward_hook)

        self.forward_features = {}
        self.backward_features = {}

    # _forward_hook is called when a pytorch module 
    # to which the hook is registered
    # the function has 3 inputs which are module, input of the module, output of the module
    def _forward_hook(self, module, input, output):
        self.forward_features['input'] = input
        self.forward_features['output'] = output
    
    # similar to the forward hook, the function has 3 inputs
    # which are module, gradient (of the loss) w.r.t. input of the module, gradient w.r.t. output of the module
    def _backward_hook(self, module, grad_input, grad_output):
        self.backward_features['grad_input'] = grad_input
        self.backward_features['grad_output'] = grad_output

    def forward(self, x, y):

        return x + y, x - y
    

if __name__ == '__main__':

    module = MyModule()
    input1 = torch.tensor([1], dtype=torch.float32, requires_grad=True)
    input2 = torch.tensor([2], dtype=torch.float32, requires_grad=True)

    # p1 = x + y
    # p2 = x - y
    pred1, pred2 = module(input1, input2)

    # l = (p1 + p2)^2 = (2x)^2 = 4x^2
    loss = torch.square(pred1 + pred2)
    loss.backward()

    ic(input1, input2, pred1, pred2)

    # look closely to the type of each parameter
    # dl/dx = 8x, dl/dy = 0, dl/dp1 = 2(p1+p2), dl/dp2 = 2(p1+p2)
    ic(module.forward_features, module.backward_features)