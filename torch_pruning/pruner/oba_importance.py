import abc
import torch
import torch.nn as nn

import typing
from . import function
from ..dependency import Group
from .importance import MagnitudeImportance
from sklearn.cluster import AffinityPropagation

class HessianImportance(MagnitudeImportance):


    def __init__(self,
                 group_reduction: str = "mean",
                 normalizer: str = 'mean',
                 multivariable: bool = False,
                 bias=False,
                 target_types: list = [nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm,
                                       nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.multivariable = multivariable
        self.target_types = target_types
        self.bias = bias
        self.p = 2

    @torch.no_grad()
    def __call__(self, group, group_importances, ch_groups=1):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue

            # Conv/Linear Output
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:

                local_imp = group_importances[layer]['output']

                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            # Conv/Linear Input
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                local_imp = group_importances[layer]['input']
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(ch_groups)
                local_imp = local_imp[:, idxs]
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            # BN
            elif prune_fn == function.prune_groupnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w * dw)
                    local_imp = torch.zeros_like(local_imp).cpu()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        db = layer.bias.grad.data[idxs]
                        local_imp = (b * db)
                        local_imp = torch.zeros_like(local_imp).cpu()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

            # LN
            elif prune_fn == function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w * dw)
                    local_imp = torch.zeros_like(local_imp).cpu()
                    # group_imp.append(local_imp)
                    # group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        db = layer.bias.grad.data[idxs]
                        local_imp = (b * db)
                        local_imp = torch.zeros_like(local_imp).cpu()
                        # group_imp.append(local_imp)
                        # group_idxs.append(root_idxs)
            assert local_imp.device == torch.device("cpu")
        if len(group_imp) == 0:  # skip groups without parameterized layers
            return None
        if self.multivariable:
            group_imp = list(map(lambda imp: imp.mean(0).abs(), group_imp))
        else:
            group_imp = list(map(lambda imp: imp.mean(0), group_imp))
        group_imp = self._reduce(group_imp, group_idxs)
        if not self.multivariable:
            group_imp = group_imp.abs()
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp