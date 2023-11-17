import torch
import torch.nn as nn

from functools import partial

from models.resnet import ResNet, resnet18


class SharpenFocus(nn.Module):

    def __init__(self, backbone_model: ResNet):
        super().__init__()

        self.model = backbone_model
        self.attention_layers = ['layer4', 'layer5']

        self.num_classes = backbone_model.num_classes
        self.parallel_last_layers = backbone_model.parallel_last_layers

        self.forward_features = {}
        self.backward_features = {}

        self._register_hooks()

    def _register_hooks(self):

        def forward_hook(name, module, input, output):
            # save output of module
            self.forward_features[name] = output

        def backward_hook(name, module, grad_input, grad_output):
            # save gradient of computed loss w.r.t. output of module
            self.backward_features[name] = grad_output[0]

        for module_name, module in self.model.named_modules():
            if module_name in self.attention_layers:
                module.register_forward_hook(partial(forward_hook, module_name))
                module.register_full_backward_hook(partial(backward_hook, module_name))

        if self.parallel_last_layers:
            for module_name, module in self.model.last_layers.items():
                module.register_forward_hook(partial(forward_hook, module_name))
                module.register_full_backward_hook(partial(backward_hook, module_name))

    def forward(self, images, labels):

        logits = self.model(images)
        self.model.zero_grad()