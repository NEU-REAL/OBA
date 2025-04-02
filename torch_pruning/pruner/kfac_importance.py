import abc
import torch
import torch.nn as nn

import typing
from . import function
from ..dependency import Group
from .importance import MagnitudeImportance
from sklearn.cluster import AffinityPropagation
class PostLayerKFACImportance(MagnitudeImportance):
    """First-order taylor expansion of the loss function.
       https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf
    """

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
                if layer in group_importances.keys():
                    if 'weight' in group_importances[layer].keys():
                        layer_weight = group_importances[layer]['weight']
                        layer_bias = group_importances[layer]['bias'] if 'bias' in group_importances[layer].keys() else None
                        if len(layer_weight.shape) == 5:
                            if self.multivariable:
                                local_imp = layer_weight.sum((3, 4)).sum(2).cpu()
                                if layer_bias is not None:
                                    local_imp += layer_bias.cpu()
                            else:
                                local_imp = layer_weight.abs().sum((3, 4)).sum(2).cpu()
                                if layer_bias is not None:
                                    local_imp += layer_bias.abs().cpu()
                        elif len(layer_weight.shape) == 3:
                            if self.multivariable:
                                local_imp = layer_weight.sum(2)
                                if layer_bias is not None:
                                    local_imp += layer_bias.cpu()
                            else:
                                local_imp = layer_weight.abs().sum(2)
                                if layer_bias is not None:
                                    local_imp += layer_bias.abs().cpu()
                        else:
                            raise Exception("Unrecognized weight shape")
                    else:
                        if self.multivariable:
                            local_imp = group_importances[layer]['output']
                        else:
                            local_imp = group_importances[layer]['output'].abs()
                    local_imp = torch.zeros_like(local_imp)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                # else:
                #     local_imp = torch.tensor([local_imp.shape[0], layer.weight.shape[0]])
                # abandon output importance
                # local_imp = torch.zeros_like(local_imp)
                # group_imp.append(local_imp)
                # group_idxs.append(root_idxs)

            # Conv/Linear Input
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if layer in group_importances.keys():
                    if 'weight' in group_importances[layer].keys():
                        layer_weight = group_importances[layer]['weight']
                        if len(layer_weight.shape) == 5:
                            if self.multivariable:
                                local_imp = layer_weight.sum((3, 4)).sum(1)
                            else:
                                local_imp = layer_weight.abs().sum((3, 4)).sum(1)
                        elif len(layer_weight.shape) == 3:
                            if self.multivariable:
                                local_imp = layer_weight.sum(1)
                            else:
                                local_imp = layer_weight.abs().sum(1)
                        else:
                            raise Exception("Unrecognized weight shape")
                    else:
                        if self.multivariable:
                            local_imp = group_importances[layer]['input']
                        else:
                            local_imp = group_importances[layer]['input'].abs()
                else:
                    local_imp = torch.zeros(local_imp.shape[0], layer.weight.shape[1])
                # repeat importance for group convolutions
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
                    local_imp = (w * dw).abs()
                    local_imp = torch.zeros_like(local_imp).cpu()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        db = layer.bias.grad.data[idxs]
                        local_imp = (b * db).abs()
                        local_imp = torch.zeros_like(local_imp).cpu()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

            # LN
            elif prune_fn == function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w * dw).abs()
                    local_imp = torch.zeros_like(local_imp).cpu()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        db = layer.bias.grad.data[idxs]
                        local_imp = (b * db).abs()
                        local_imp = torch.zeros_like(local_imp).cpu()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            # assert local_imp.device == torch.device("cpu")
        if len(group_imp) == 0:  # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        if self.multivariable:
            group_imp = group_imp.mean(0)
        else:
            group_imp = group_imp.mean(0)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp