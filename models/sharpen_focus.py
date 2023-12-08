import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.transforms.functional import to_pil_image
from PIL import Image
from functools import partial

from models.resnet import ResNet, resnet18


class SharpenFocus(nn.Module):

    def __init__(
            self,
            backbone_model: ResNet,
            sigma=0.55,
            omega=100,
            theta=0.8
        ):
        super().__init__()

        self.model = backbone_model
        self.attention_layers = ('layer4', 'layer5')

        self.num_classes = backbone_model.num_classes
        self.parallel_block_channels = backbone_model.parallel_block_channels
        
        self.sigma = sigma
        self.omega = omega
        self.theta = theta

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

        if self.parallel_block_channels:
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
        confused_indices = torch.clone(first_indices)  # index of wrong or confused class of each sample from the batch
        confused_indices[correct_pred_sample_indices] = second_indices[correct_pred_sample_indices]

        # compute attentions w.r.t. confused logit scores (confused class)
        A_confused_inner, A_confused_last, _, _ = self._get_inner_last_attention_map(logits, confused_indices)

        # compute attentions w.r.t. label logit scores (true class)
        A_true_inner, A_true_last, ff_true, _ = self._get_inner_last_attention_map(logits, labels)

        as_in_loss, _ = self._compute_as_loss(A_true_inner, A_confused_inner, sigma=self.sigma, omega=self.omega)
        as_la_loss, la_mask = self._compute_as_loss(A_true_last, A_confused_last, sigma=self.sigma, omega=self.omega)
        
        resized_la_mask = F.interpolate(la_mask, size=A_true_inner.shape[-2:], mode='bilinear', align_corners=False)
        ac_loss = self._compute_ac_loss(A_true_inner, resized_la_mask, theta=self.theta)

        if self.parallel_block_channels:
            bw_loss = self._compute_black_white_loss(labels, ff_true)
            return logits, A_true_last, A_confused_last, ac_loss, as_in_loss, as_la_loss, bw_loss
        else:
            return logits, A_true_last, A_confused_last, ac_loss, as_in_loss, as_la_loss

    def _populate_gradient(self, logits, labels):
        '''Backpropagate gradient only through each last layer designated by label, return forward/backward features.'''

        masked_logit = torch.sum(logits * F.one_hot(labels, num_classes=self.num_classes))
        masked_logit.backward(gradient=masked_logit, retain_graph=True)
        self.model.zero_grad()

        return self.forward_features, self.backward_features

    def _get_module_attention_map(self, module_name, forward_features, backward_features):
        '''Get attention map of a single module from backward/forward features dictionaries.'''

        grad_activation = backward_features[module_name]
        activation = forward_features[module_name]
        # grad_activation.shape, activation.shape: (batch_size, out_channels, height, width)

        weights = F.adaptive_avg_pool2d(F.relu(grad_activation), 1)  # weights.shape: (batch_size, out_channels, 1, 1)
        attention_map = F.relu(torch.sum(torch.mul(activation, weights), 1, keepdim=True))  # attention_map.shape: (batch_size, 1, height, width)

        return attention_map
    
    def _get_inner_last_attention_map(self, logits, labels):
        '''Get inner and last layers' attention maps, return the maps and features from which the maps are computed.'''

        forward_features, backward_features = self._populate_gradient(logits, labels)

        inner_layer_attention_map = \
            self._get_module_attention_map(self.attention_layers[0], forward_features, backward_features)
        
        if not self.parallel_block_channels:
            last_layer_attention_map = \
                self._get_module_attention_map(self.attention_layers[1], forward_features, backward_features)
        else:
            maps = []
            for idx in range(len(labels)):
                module_name = 'layer5class' + str(labels[idx].item())
                map = self._get_module_attention_map(module_name, forward_features, backward_features)[idx]
                maps.append(map)
            last_layer_attention_map = torch.stack(maps)

        return inner_layer_attention_map, last_layer_attention_map, forward_features, backward_features

    def _compute_black_white_loss(self, labels, forward_features):

        mean_features = []
        for idx in range(self.num_classes):
            module_name = 'layer5class' + str(idx)
            mean_features.append(torch.mean(forward_features[module_name], dim=(1, 2, 3)))
        mean_features = torch.stack(mean_features, dim=1)  # (batch_size, num_classes)

        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes)
        mean_features_true = torch.sum(mean_features * labels_one_hot, dim=1)  # activation of true (1) class
        mean_features_false = torch.sum(mean_features * (1 - labels_one_hot), dim=1) / (self.num_classes - 1)  # mean activation of false (200 - 1) class

        bw_loss = torch.mean(torch.exp(mean_features_false - mean_features_true))

        return bw_loss

    def _compute_as_loss(self, A_true, A_confused, sigma=0.55, omega=100):

        eps = 1e-6  # for numerical stability

        # if sigmoid is going to be used to compute the attention map,
        # scaling should be modified to account for range change
        # for now, using relu makes the range of attention map (0, A_max)
        # using sigmoid makes it (A_min, A_max)

        with torch.no_grad():
            A_true_max = torch.max(A_true)

        scaled_A_true = A_true / (A_true_max + eps)
        mask = torch.sigmoid(omega * (scaled_A_true - sigma))

        num = torch.sum(torch.min(A_true, A_confused) * mask)
        den = torch.sum(A_true + A_confused) + eps

        as_loss = 2 * num / den

        return as_loss, mask
    
    def _compute_ac_loss(self, A_true, mask, theta=0.8):

        eps = 1e-6  # for numerical stability

        num = torch.sum(A_true * mask)
        den = torch.sum(A_true) + eps
        
        ac_loss = theta - num / den

        return ac_loss


def sfocus18(num_classses, sigma=0.55, omega=100, theta=0.8, parallel_block_channels=0):

    backbone = resnet18(num_classses, parallel_block_channels)
    model = SharpenFocus(backbone, sigma=sigma, omega=omega, theta=theta)

    return model

def visualize_attention(images, attention_maps, unorm):

    batch_size = images.shape[0]
    image_h, image_w = images.shape[2:]
    map_h, map_w = attention_maps.shape[2:]

    overlays = []
    for idx in range(batch_size):
        map = torch.cat((attention_maps[idx], torch.zeros(2, map_h, map_w)))
        pil_map = to_pil_image(map).resize((image_h, image_w))
        pil_image = to_pil_image(unorm(images[idx]))
        np_overlay = np.asarray(Image.blend(pil_image, pil_map, 0.5))
        overlay = torch.permute(torch.tensor(np_overlay), (2, 0, 1))
        overlays.append(overlay)

    return torch.stack(overlays)

def tensor_test():

    from icecream import ic

    num_classes = 3
    parallel_block_channels = True
    backbone = resnet18(num_classes, parallel_block_channels)
    model = SharpenFocus(backbone)
    
    batch_size = 4
    num_channels = 3
    width = 2
    height = 2

    images = torch.rand((batch_size, num_channels, width, height))
    labels = torch.randint(0, num_classes, (batch_size,))
    
    ic(model)
    ic(images, labels)
    ic(model(images, labels))

def dataset_test():

    import PIL
    import matplotlib.pyplot as plt

    from icecream import ic
    from utils.data_utils import get_datasets
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid

    train_dataset, val_dataset, num_classes, unorm = get_datasets('TinyImageNet')

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    parallel_block_channels = 1
    backbone = resnet18(num_classes, parallel_block_channels)
    model = SharpenFocus(backbone)
    ic(model)
    
    for idx, (images, labels) in enumerate(val_loader):
    
        logits, A_true_la, A_conf_la, ac_loss, as_in_loss, as_la_loss, bw_loss = model(images, labels)
        ic(labels, ac_loss, as_in_loss, as_la_loss, bw_loss)

        overlays = visualize_attention(images, A_true_la, unorm)
        grid = make_grid(overlays).permute(1, 2, 0)
        plt.imshow(grid)
        plt.show()

        if idx == 2: break
