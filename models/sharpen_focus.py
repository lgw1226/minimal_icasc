import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from models.resnet import ResNet, resnet18


class SharpenFocus(nn.Module):

    def __init__(self, backbone_model: ResNet):
        super().__init__()

        self.model = backbone_model
        self.attention_layers = ('layer4', 'layer5')

        self.num_classes = backbone_model.num_classes
        self.parallel_last_layers = backbone_model.parallel_last_layers

        self.forward_features = {}
        self.backward_features = {}

        self._register_hooks()

    def _register_hooks(self):

        def forward_hook(name, module, input, output):
            # save output (activation) after passing input through module
            self.forward_features[name] = output

        def backward_hook(name, module, grad_input, grad_output):
            # save gradient of computed loss w.r.t. output (activation) of module
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

        top2_values, top2_indices = torch.topk(logits, 2)
        first_indices = top2_indices[:,0]
        second_indices = top2_indices[:,1]

        correct_pred_sample_indices = torch.nonzero((first_indices==labels)==True)
        confused_indices = torch.clone(first_indices)  # index of wrong or confused class of each sample
        confused_indices[correct_pred_sample_indices] = second_indices[correct_pred_sample_indices]

        self._populate_gradient(logits, confused_indices)
        inner_attention_map = self._get_attention_map(self.attention_layers[0])
        if not self.parallel_last_layers:
            last_attention_map = self._get_attention_map(self.attention_layers[1])
        else: pass
            # case: top-1 prediction being wrong and is included in confused indices
        return first_indices, second_indices, correct_pred_sample_indices, confused_indices
    
    def _populate_gradient(self, logits, labels):
        '''Backpropagate gradient only through each last layer designated by label.'''

        masked_mean_logit = torch.mean(logits[labels])  # logits[labels].shape: (batch_size,)
        masked_mean_logit.backward(retain_graph=True)
        self.model.zero_grad()

    def _get_attention_map(self, module_name):

        grad_activation = self.backward_features[module_name]
        activation = self.forward_features[module_name]
        # grad_activation.shape, activation.shape: (batch_size, out_channels, width, height)

        weights = F.adaptive_avg_pool2d(F.relu(grad_activation), 1)  # weights.shape: (batch_size, out_channels, 1, 1)
        attention_map = F.relu(torch.sum(torch.mul(activation, weights), 1, keepdim=True))  # attention_map.shape: (batch_size, 1, width, height)

        return attention_map


def test():

    from icecream import ic


    num_classes = 3
    parallel_last_layers = True
    backbone = resnet18(num_classes, parallel_last_layers)
    model = SharpenFocus(backbone)
    
    batch_size = 4
    num_channels = 3
    width = 2
    height = 2

    images = torch.rand((batch_size, num_channels, width, height))
    labels = torch.randint(0, num_classes, (batch_size,))  # ImageNet labels are given as ids
    
    ic(model)
    ic(images, labels)
    ic(model(images, labels))