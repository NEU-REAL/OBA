import torch
import torch.nn as nn
import typing, warnings
from .metapruner import MetaPruner
from .scheduler import linear_scheduler, exponential_scheduler
from .. import function
from ..._helpers import _FlattenIndexMapping
from ... import ops
from modules.models import Attention, forward_hook, backward_hook
from torch.nn.utils import prune
from torch.backends.cuda import sdp_kernel
import time
import torch.distributed as dist
from tqdm import tqdm
def factorial(n):
    assert isinstance(n, int), "n should be an integer"
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def safe_div(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    # Replace values in denom that are close to zero with eps
    safe_denom = torch.where(denom != 0, denom, 1.0)
    return numer / safe_denom


class FastOBAPruner(MetaPruner):
    """Optimal Brain Apoptosis
    """

    def __init__(
            self,
            model: nn.Module,  # a simple pytorch model
            example_inputs: torch.Tensor,  # a dummy input for graph tracing. Should be on the same
            importance: typing.Callable,  # tp.importance.Importance for group importance estimation
            reg=1e-4,  # regularization coefficient
            alpha=4,  # regularization scaling factor, [2^0, 2^alpha]
            global_pruning: bool = False,
            # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
            pruning_ratio: float = 0.5,  # channel/dim pruning ratio, also known as pruning ratio
            pruning_ratio_dict: typing.Dict[nn.Module, float] = None,
            # layer-specific pruning ratio, will cover pruning_ratio if specified
            max_pruning_ratio: float = 0.95,  # maximum pruning ratio. useful if over-pruning happens.
            iterative_steps: int = 1,  # for iterative pruning
            iterative_pruning_ratio_scheduler: typing.Callable = exponential_scheduler,  # scheduler for iterative pruning.
            ignored_layers: typing.List[nn.Module] = None,  # ignored layers
            round_to: int = None,  # round channels to the nearest multiple of round_to

            # Advanced
            in_channel_groups: typing.Dict[nn.Module, int] = dict(),  # The number of channel groups for layer input
            out_channel_groups: typing.Dict[nn.Module, int] = dict(),  # The number of channel groups for layer output
            num_heads: typing.Dict[nn.Module, int] = dict(),  # The number of heads for multi-head attention
            prune_num_heads: bool = False,  # remove entire heads in multi-head attention
            prune_head_dims: bool = True,  # remove head dimensions in multi-head attention
            head_pruning_ratio: float = 0.0,  # head pruning ratio
            head_pruning_ratio_dict: typing.Dict[nn.Module, float] = None,  # layer-specific head pruning ratio
            customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None,
            # pruners for customized layers. E.g., {nn.Linear: my_linear_pruner}
            unwrapped_parameters: typing.Dict[nn.Parameter, int] = None,
            # unwrapped nn.Parameters & pruning_dims. For example, {ViT.pos_emb: 0}
            root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],
            # root module for each group
            forward_fn: typing.Callable = None,  # a function to execute model.forward
            output_transform: typing.Callable = None,  # a function to transform network outputs

            # deprecated
            channel_groups: typing.Dict[nn.Module, int] = dict(),  # channel groups for layers
            ch_sparsity: float = None,
            ch_sparsity_dict: typing.Dict[nn.Module, float] = None,

            # self-defined
            record_length: int = 1000,
            delta:float = 1.0,
            sl_lr = 1e-5,
    ):
        super(FastOBAPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            global_pruning=global_pruning,
            pruning_ratio=pruning_ratio,
            pruning_ratio_dict=pruning_ratio_dict,
            max_pruning_ratio=max_pruning_ratio,
            iterative_steps=iterative_steps,
            iterative_pruning_ratio_scheduler=iterative_pruning_ratio_scheduler,
            ignored_layers=ignored_layers,
            round_to=round_to,

            in_channel_groups=in_channel_groups,
            out_channel_groups=out_channel_groups,
            num_heads=num_heads,
            prune_num_heads=prune_num_heads,
            prune_head_dims=prune_head_dims,
            head_pruning_ratio=head_pruning_ratio,
            head_pruning_ratio_dict=head_pruning_ratio_dict,
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
            root_module_types=root_module_types,
            forward_fn=forward_fn,
            output_transform=output_transform,

            channel_groups=channel_groups,
            ch_sparsity=ch_sparsity,
            ch_sparsity_dict=ch_sparsity_dict
        )

        self.reg = reg
        self.alpha = alpha
        self._groups = list(
            self.DG.get_all_groups(root_module_types=self.root_module_types, ignored_layers=self.ignored_layers))
        self.cnt = 0
        self.group_importances = self.initialize_importance()
        self.record_length = record_length
        self.init_target_parameters()
        self.module2parameters = self.initialize_module2parameters_mapping(self.target_parameters)
        self.parameters2module = self.initialize_parameters2module_mapping(self.target_parameters)
        self.delta = delta
        self.sl_lr = sl_lr

    def step(self, interactive=False):
        self.current_step += 1
        pruning_method = self.prune_global if self.global_pruning else self.prune_local

        if interactive:  # yield groups for interactive pruning
            return pruning_method()
        else:
            for group in pruning_method():
                group.prune()

    # unstructured pruning for the model
    def unstructured_prune(self, pruning_ratio):
        def _normalize(group_importance, normalizer):
            if normalizer is None or normalizer == "none":
                return group_importance
            elif isinstance(normalizer, typing.Callable):
                return normalizer(group_importance)
            elif normalizer == "sum":
                return safe_div(group_importance, group_importance.sum())
            elif normalizer == "standarization":
                return safe_div(group_importance - group_importance.min(), group_importance.max() - group_importance.min())
            elif normalizer == "mean":
                return safe_div(group_importance, group_importance.mean())
            elif normalizer == "max":
                return safe_div(group_importance, group_importance.max())
            elif normalizer == 'gaussian':
                return safe_div(group_importance - group_importance.mean(), group_importance.std())
            elif normalizer == "norm":
                return safe_div(group_importance, group_importance.norm())

        global_importance = []
        separate_importance = {}
        param_num = 0
        mask_dict = {}
        max_weight = torch.cat([param.data.abs().flatten() for param in self.model.parameters()]).max()
        for module in self.model.modules():
            if module in self.group_importances.keys():
                weight_importance = self.group_importances[module]['weight'].abs() / self.group_importances[module]['count']
                ### You can manually change the code here to use different importance types
                # weight_importance = weight_importance ** 2
                # weight_importance = self.group_importances[module]['weight'] / self.group_importances[module]['count']
                # weight_importance = module.weight.data.abs()
                param_num += weight_importance.numel()
                importance = _normalize(weight_importance, self.importance.normalizer)
                importance += module.weight.data.abs() / max_weight # resolve gradient vanishing
                separate_importance[module] = {}
                separate_importance[module]['weight'] = importance
                importance = importance.view(-1)
                if 'bias' in self.group_importances[module].keys():
                    bias_importance = self.group_importances[module]['bias'].abs().mean(0).to(module.bias.device)  / self.group_importances[module]['count']
                    ### You can manually change the code here to use different importance types
                    # bias_importance = bias_importance ** 2
                    # bias_importance = self.group_importances[module]['bias'] / self.group_importances[module]['count']
                    # bias_importance = module.bias.data.abs()
                    bias_importance = _normalize(bias_importance, self.importance.normalizer).view(-1)
                    bias_importance += module.bias.data.abs() / max_weight  # resolve gradient vanishing
                    separate_importance[module]['bias'] = bias_importance
                    importance = torch.cat([importance, bias_importance])
                    param_num += bias_importance.numel()
                global_importance.append(importance)
        global_importance = torch.cat(global_importance)
        n_pruned = int(pruning_ratio * param_num)
        topk_imp, _ = torch.topk(global_importance, k=n_pruned, largest=True)
        thresh = topk_imp[-1]
        # update parameters
        for module in self.model.modules():
            if module in self.group_importances.keys():
                mask = separate_importance[module]['weight'] > thresh
                prune.custom_from_mask(module, name='weight', mask=mask)
                mask_dict[module] = {}
                mask_dict[module]['weight'] = mask
                if 'bias' in self.group_importances[module].keys():
                    mask = separate_importance[module]['bias'] > thresh
                    prune.custom_from_mask(module, name='bias', mask=mask)
                    mask_dict[module]['bias'] = mask
        return mask_dict

    def layerwise_unstructured_prune(self, pruning_ratio):
        def _normalize(group_importance, normalizer):
            if normalizer is None or normalizer == "none":
                return group_importance
            elif isinstance(normalizer, typing.Callable):
                return normalizer(group_importance)
            elif normalizer == "sum":
                return group_importance / (group_importance.sum() + 1e-6)
            elif normalizer == "standarization":
                return (group_importance - group_importance.min()) / (
                            group_importance.max() - group_importance.min() + 1e-8)
            elif normalizer == "mean":
                return group_importance / (group_importance.mean() + 1e-6)
            elif normalizer == "max":
                return group_importance / (group_importance.max() + 1e-6)
            elif normalizer == 'gaussian':
                return (group_importance - group_importance.mean()) / (group_importance.std() + 1e-8)
            elif normalizer == "norm":
                return group_importance / (group_importance.norm() + 1e-6)

        global_importance = []
        separate_importance = {}
        param_num = 0
        mask_dict = {}
        max_weight = torch.cat([param.data.abs().flatten() for param in self.model.parameters()]).max()
        for module in self.model.modules():
            if module in self.group_importances.keys() and module not in self.ignored_layers:
                weight_importance = self.group_importances[module]['weight'].abs() / self.group_importances[module]['count']
                ### You can manually change the code here to use different importance types
                # weight_importance = weight_importance ** 2
                # weight_importance = self.group_importances[module]['weight'] / self.group_importances[module]['count']
                # weight_importance = module.weight.data.abs()
                param_num += weight_importance.numel()
                importance = _normalize(weight_importance, self.importance.normalizer)
                max_weight = module.weight.data.abs().max()
                importance += module.weight.data.abs() / max_weight
                separate_importance[module] = {}
                separate_importance[module]['weight'] = importance
                importance = importance.view(-1)
                if 'bias' in self.group_importances[module].keys():
                    bias_importance = self.group_importances[module]['bias'].abs().mean(0).to(module.bias.device)  / self.group_importances[module]['count']
                    ### You can manually change the code here to use different importance types
                    # bias_importance = bias_importance ** 2
                    # bias_importance = self.group_importances[module]['bias'] / self.group_importances[module]['count']
                    # bias_importance = module.bias.data.abs()
                    bias_importance = _normalize(bias_importance, self.importance.normalizer).view(-1)
                    max_weight = module.bias.data.abs().max()
                    bias_importance += module.bias.data.abs() / max_weight
                    separate_importance[module]['bias'] = bias_importance
                    importance = torch.cat([importance, bias_importance])
                    param_num += bias_importance.numel()
                global_importance.append(importance)
        global_importance = torch.cat(global_importance)
        n_pruned = int(pruning_ratio * param_num)
        topk_imp, _ = torch.topk(global_importance, k=n_pruned, largest=True)
        # update parameters
        for module in self.model.modules():
            if module in self.group_importances.keys() and module not in self.ignored_layers:
                topk_imp, _ = torch.topk(separate_importance[module]['weight'].view(-1), k=int(pruning_ratio * module.weight.data.numel()), largest=True)
                thresh = topk_imp[-1]
                # prune.remove(module, name='weight')
                mask = separate_importance[module]['weight'] > thresh
                prune.custom_from_mask(module, name='weight', mask=mask)
                mask_dict[module] = {}
                mask_dict[module]['weight'] = mask
                if 'bias' in self.group_importances[module].keys():
                    topk_imp, _ = torch.topk(separate_importance[module]['bias'].view(-1),
                                             k=int(pruning_ratio * module.bias.data.numel()), largest=True)
                    thresh = topk_imp[-1]
                    mask = separate_importance[module]['bias'] > thresh
                    # prune.remove(module, name='bias')
                    prune.custom_from_mask(module, name='bias', mask=mask)
                    mask_dict[module]['bias'] = mask
        return mask_dict

    def estimate_importance(self, group, ch_groups=1) -> torch.Tensor:
        return self.importance(group, self.group_importances, ch_groups=ch_groups)
        # return self.importance(group, ch_groups=ch_groups)

    def update_dependency_graph(self, example_inputs):
        self.DG = self.DG.build_dependency(
            self.model,
            example_inputs=example_inputs,
            forward_fn=None,
            output_transform=None,
            unwrapped_parameters=[],
            customized_pruners=None,
            ignored_params=self.ignored_params,
        )

    def any_order_differentiation(self, loss, delta=-1.0, parameters=None, order=1, parameter_mask_dict=None, distributed=False):
        if parameters is None:
            parameters = self.model.parameters()
        grads = [torch.zeros_like(param) for param in parameters]
        for current_order in range(1, order + 1):
            if current_order == 1:
                if current_order == order:
                    current_grad = torch.autograd.grad(loss, parameters)
                else:
                    current_grad = torch.autograd.grad(loss, parameters, create_graph=True)
            else:
                if parameter_mask_dict is not None:
                    grad_outputs = [param * delta * mask for param, mask in zip(parameters, parameter_mask_dict)]
                else:
                    grad_outputs = [param * delta for param in parameters]
                if current_order == order:
                    current_grad = torch.autograd.grad(current_grad, parameters,
                                                       grad_outputs=grad_outputs)
                else:
                    current_grad = torch.autograd.grad(current_grad, parameters,
                                                       grad_outputs=grad_outputs, create_graph=True)
        grads = [current_grad.detach() * param.data * delta for grad, current_grad, param in zip(grads, current_grad, parameters)]
        # if in DDP mode, all‐reduce and average the grads
        if distributed:
            world_size = dist.get_world_size()
            for i, g in enumerate(grads):
                dist.all_reduce(g, op=dist.ReduceOp.SUM)
                grads[i] = g.div(world_size)
        return grads

    def obtain_importance(self, loss, order=2, distributed=False):
        current_group_importances = self.initialize_importance()
        grads = self.any_order_differentiation(loss, delta=self.delta, parameters=self.target_parameters, order=order, distributed=distributed)
        for module in current_group_importances.keys():
            if "weight" in self.module2parameters[module].keys():
                current_group_importances[module]["weight"] = grads[self.module2parameters[module]["weight"]]
            if "bias" in self.module2parameters[module].keys():
                current_group_importances[module]["bias"] = grads[self.module2parameters[module]["bias"]]
        if distributed:
            import torch.distributed as dist
            world_size = dist.get_world_size()
            for module, imps in current_group_importances.items():
                for name, tensor in imps.items():
                    # all_reduce in‐place sum, then divide to get the mean
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    tensor.div_(world_size)
                    imps[name] = tensor
        self.update_group_importance(current_group_importances)

    def obtain_sparse_importance(self, loss, parameter_mask_dict, order=2):
        current_group_importances = self.initialize_importance()
        grads = self.any_order_differentiation(loss, delta=self.delta, parameters=self.target_parameters, order=order, parameter_mask_dict=parameter_mask_dict, multiply_param=False)
        for module in current_group_importances.keys():
            if "weight" in self.module2parameters[module].keys():
                current_group_importances[module]["weight"] = grads[self.module2parameters[module]["weight"]]
            if "bias" in self.module2parameters[module].keys():
                current_group_importances[module]["bias"] = grads[self.module2parameters[module]["bias"]]
        self.update_group_importance(current_group_importances)

    def update_group_importance(self, current_group_importances):
        ## sum importances from weights and biases
        ## update the group importance
        for module in self.model.modules():
            if module in self.group_importances:
                if hasattr(module, "weight"):
                    layer_weight = current_group_importances[module]['weight']
                    if len(layer_weight.shape) == 4:
                        input_imp = layer_weight.sum((0, 2, 3))
                        output_imp = layer_weight.sum((1, 2, 3))
                    elif len(layer_weight.shape) == 2:
                        input_imp = layer_weight.sum(0)
                        output_imp = layer_weight.sum(1)
                    elif len(layer_weight.shape) == 1:
                        input_imp = layer_weight
                        output_imp = layer_weight
                    else:
                        raise Exception("Unrecognized layer weight shape")
                    layer_bias = current_group_importances[module]['bias'] if 'bias' in current_group_importances[module].keys() else None
                    # if layer_bias is not None:
                    #     output_imp += layer_bias
                    self.group_importances[module]["input"] = torch.cat(
                        [self.group_importances[module]["input"], input_imp.unsqueeze(0).cpu()])
                    if self.group_importances[module]["input"].shape[0] > self.record_length:
                        self.group_importances[module]["input"] = self.group_importances[module]["input"][1:]
                    self.group_importances[module]["output"] = torch.cat(
                        [self.group_importances[module]["output"], output_imp.unsqueeze(0).cpu()])
                    if self.group_importances[module]["output"].shape[0] > self.record_length:
                        self.group_importances[module]["output"] = self.group_importances[module]["output"][1:]
                    if layer_bias is not None:
                        self.group_importances[module]["bias"] = torch.cat(
                            [self.group_importances[module]["bias"], layer_bias.unsqueeze(0).cpu()])
                        if self.group_importances[module]["bias"].shape[0] > self.record_length:
                            self.group_importances[module]["bias"] = self.group_importances[module]["bias"][1:]


                    # update weight importance
                    self.group_importances[module]["weight"] += layer_weight.to(torch.float64)
                    self.group_importances[module]["count"] += 1
                    # if layer_bias is not None:
                    #     self.group_importances[module]["bias"] += layer_bias.to(torch.float64)


                    # self.group_importances[module]["weight"] = torch.cat(
                    #     [self.group_importances[module]["weight"],
                    #      current_group_importances[module]["weight"].unsqueeze(0).cpu()])
                    # if self.group_importances[module]["weight"].shape[0] > self.record_length:
                    #     self.group_importances[module]["weight"] = self.group_importances[module]["weight"][1:]
                #     self.group_importances[module]["first_order_weight"] = torch.cat(
                #         [self.group_importances[module]["first_order_weight"],
                #          current_group_importances[module]["first_order_weight"].unsqueeze(0).cpu()])
                #     if self.group_importances[module]["first_order_weight"].shape[0] > self.record_length:
                #         self.group_importances[module]["first_order_weight"] = self.group_importances[module]["first_order_weight"][1:]
                # if hasattr(module, "bias") and module.bias is not None:
                #     self.group_importances[module]["bias"] = torch.cat(
                #         [self.group_importances[module]["bias"],
                #          current_group_importances[module]["bias"].unsqueeze(0).cpu()])
                #     if self.group_importances[module]["bias"].shape[0] > self.record_length:
                #         self.group_importances[module]["bias"] = self.group_importances[module]["bias"][1:]
                #     self.group_importances[module]["first_order_bias"] = torch.cat(
                #         [self.group_importances[module]["first_order_bias"],
                #          current_group_importances[module]["first_order_bias"].unsqueeze(0).cpu()])
                #     if self.group_importances[module]["first_order_bias"].shape[0] > self.record_length:
                #         self.group_importances[module]["first_order_bias"] = self.group_importances[module]["first_order_bias"][1:]


    def initialize_importance(self):
        group_importances = {}
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and not isinstance(module, torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
                group_importances[module] = {}
                if hasattr(module, "weight"):
                    group_importances[module]["weight"] = torch.zeros_like(module.weight.data).to(torch.float64)
                # group_importances[module]["first_order_weight"] = torch.tensor([])
                if hasattr(module, "bias") and module.bias is not None:
                    group_importances[module]["bias"] = torch.tensor([])#torch.zeros_like(module.bias.data).to(torch.float64)
                    # group_importances[module]["first_order_bias"] = torch.tensor([])
                group_importances[module]["input"] = torch.tensor([])
                group_importances[module]["output"] = torch.tensor([])
                group_importances[module]["count"] = 0
            # elif isinstance(module, nn.BatchNorm2d):
            #     group_importances[module] = {}
            #     group_importances[module]["weight"] = torch.zeros_like(module.weight.data).to(torch.float64)
            #     group_importances[module]["bias"] = torch.zeros_like(module.bias.data).to(torch.float64)
            #     group_importances[module]["input"] = torch.tensor([])
            #     group_importances[module]["output"] = torch.tensor([])
            #     group_importances[module]["count"] = 0
        return group_importances

    def init_target_parameters(self):
        target_parameters = []
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention, nn.BatchNorm2d, nn.LayerNorm)) and not isinstance(module, torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
                target_parameters.extend(module.parameters())
        self.target_parameters = target_parameters
        self.target_parameters = list(self.model.parameters())

    def initialize_module2parameters_mapping(self, parameters=None):
        module2parameters = {}
        if parameters is None:
            parameters = self.model.parameters()
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, torch.nn.MultiheadAttention, Attention, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm, nn.LayerNorm)) and not isinstance(module, torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
                module2parameters[module] = {}
                if hasattr(module, "weight"):
                    for i, param in enumerate(parameters):
                        if module.weight is param:
                            module2parameters[module]["weight"] = i
                            break
                if hasattr(module, "bias") and module.bias is not None:
                    for i, param in enumerate(parameters):
                        if module.bias is param:
                            module2parameters[module]["bias"] = i
                            break
        return module2parameters

    def initialize_parameters2module_mapping(self, parameters=None):
        module2parameters = self.initialize_module2parameters_mapping(parameters)

        parameters2module = {}
        for module, name2idx in module2parameters.items():
            for attr_name, idx in name2idx.items():
                parameters2module[idx] = (module, attr_name)

        return parameters2module


    def prune_local(self) -> typing.Generator:
        if self.current_step > self.iterative_steps:
            warnings.warn("Pruning exceed the maximum iterative steps, no pruning will be performed.")
            return

        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers,
                                            root_module_types=self.root_module_types):
            if self._check_pruning_ratio(group):  # check pruning ratio
                ##################################
                # Compute raw importance score
                ##################################
                group = self._downstream_node_as_root_if_attention(group)
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler
                ch_groups = self._get_channel_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups)
                if imp is None: continue

                ##################################
                # Compute the number of dims/channels to prune
                ##################################
                if self.DG.is_out_channel_pruning_fn(pruning_fn):
                    current_channels = self.DG.get_out_channels(module)
                    target_pruning_ratio = self.get_target_pruning_ratio(module)
                    n_pruned = current_channels - int(
                        self.layer_init_out_ch[module] *
                        (1 - target_pruning_ratio)
                    )
                else:
                    current_channels = self.DG.get_in_channels(module)
                    target_pruning_ratio = self.get_target_pruning_ratio(module)
                    n_pruned = current_channels - int(
                        self.layer_init_in_ch[module] *
                        (1 - target_pruning_ratio)
                    )
                # round to the nearest multiple of round_to
                if self.round_to:
                    n_pruned = self._round_to(n_pruned, current_channels, self.round_to)

                ##################################
                # collect pruning idxs
                ##################################
                pruning_idxs = []
                _is_attn, qkv_layers = self._is_attn_group(group)
                group_size = current_channels // ch_groups
                # dims/channels
                if n_pruned > 0:
                    if (self.prune_head_dims and _is_attn) or (not _is_attn):
                        n_pruned_per_group = n_pruned // ch_groups
                        if self.round_to:
                            n_pruned_per_group = self._round_to(n_pruned_per_group, group_size, self.round_to)
                        if n_pruned_per_group > 0:
                            for chg in range(ch_groups):
                                sub_group_imp = imp[chg * group_size: (chg + 1) * group_size]
                                sub_imp_argsort = torch.argsort(sub_group_imp)
                                sub_pruning_idxs = sub_imp_argsort[:n_pruned_per_group] + chg * group_size  # offset
                                pruning_idxs.append(sub_pruning_idxs)
                else:  # no channel grouping
                    imp_argsort = torch.argsort(imp)
                    pruning_idxs.append(imp_argsort[:n_pruned])

                # num heads
                if _is_attn and self.prune_num_heads:  # Prune entire attn heads
                    target_head_pruning_ratio = self.get_target_head_pruning_ratio(qkv_layers[0])
                    n_heads_removed = self.num_heads[qkv_layers[0]] - int(
                        self.init_num_heads[qkv_layers[0]] * (1 - target_head_pruning_ratio))
                    if n_heads_removed > 0:
                        head_imp = imp.view(ch_groups, -1).mean(1)
                        for head_id in torch.argsort(head_imp)[:n_heads_removed]:
                            pruning_idxs.append(
                                torch.arange(head_id * group_size, (head_id + 1) * group_size, device=head_imp.device))

                if len(pruning_idxs) == 0: continue
                pruning_idxs = torch.unique(torch.cat(pruning_idxs, 0)).tolist()
                group = self.DG.get_pruning_group(
                    module, pruning_fn, pruning_idxs)

                if self.DG.check_pruning_group(group):
                    # Update num heads after pruning
                    if _is_attn and self.prune_num_heads and n_heads_removed > 0:
                        for dep, _ in group:
                            if dep.target.module in self.num_heads:
                                self.num_heads[dep.target.module] -= n_heads_removed
                    yield group

    def prune_global(self) -> typing.Generator:
        if self.current_step > self.iterative_steps:
            warnings.warn("Pruning exceed the maximum iterative steps, no pruning will be performed.")
            return

        ##############################################
        # 1. Pre-compute importance for each group
        ##############################################
        global_importance = []
        global_head_importance = {}  # for attn head pruning
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers,
                                            root_module_types=self.root_module_types):
            if self._check_pruning_ratio(group):
                group = self._downstream_node_as_root_if_attention(
                    group)  # use a downstream node as the root node for attention layers
                ch_groups = self._get_channel_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups)  # raw importance score
                # imp = torch.rand(imp.shape)
                group_size = len(imp) // ch_groups
                if imp is None: continue
                if ch_groups > 1:
                    # average importance across groups. For example:
                    # imp = [1, 2, 3, 4, 5, 6] with ch_groups=2
                    # We have two groups [1,2,3] and [4,5,6]
                    # the average importance is [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
                    # since some
                    dim_imp = imp.view(ch_groups, -1).mean(dim=0)
                else:
                    # no grouping
                    dim_imp = imp
                global_importance.append((group, ch_groups, group_size, dim_imp))

                # pre-compute head importance for attn heads
                _is_attn, qkv_layers = self._is_attn_group(group)
                if _is_attn and self.prune_num_heads and self.get_target_head_pruning_ratio(qkv_layers[0]) > 0:
                    # average importance of each group. For example:
                    # the importance score of the group
                    # imp = [1, 2, 3, 4, 5, 6] with num_heads=2
                    # Note: head1 = [1, 2, 3], head2 = [4, 5, 6]
                    # the average importance is [(1+2+3)/3, (4+5+6)/3] = [2, 5]
                    head_imp = imp.view(ch_groups, -1).mean(1)  # average importance by head.
                    global_head_importance[group] = (qkv_layers, head_imp)

        if len(global_importance) == 0 and len(global_head_importance) == 0:
            return

        ##############################################
        # 2. Thresholding by concatenating all importance scores
        ##############################################

        # Find the threshold for global pruning
        if len(global_importance) > 0:
            concat_imp = torch.cat([local_imp[-1] for local_imp in global_importance], dim=0)
            target_pruning_ratio = self.per_step_pruning_ratio[self.current_step]
            n_pruned = len(concat_imp) - int(
                self.initial_total_channels *
                (1 - target_pruning_ratio)
            )
            if n_pruned > 0:
                topk_imp, _ = torch.topk(concat_imp, k=n_pruned, largest=False)
                thres = topk_imp[-1]

        # Find the threshold for head pruning
        if len(global_head_importance) > 0:
            concat_head_imp = torch.cat([local_imp[-1] for local_imp in global_head_importance.values()], dim=0)
            target_head_pruning_ratio = self.per_step_head_pruning_ratio[self.current_step]
            n_heads_removed = len(concat_head_imp) - int(
                self.initial_total_heads *
                (1 - target_head_pruning_ratio)
            )
            if n_heads_removed > 0:
                topk_head_imp, _ = torch.topk(concat_head_imp, k=n_heads_removed, largest=False)
                head_thres = topk_head_imp[-1]

        ##############################################
        # 3. Prune
        ##############################################
        for group, ch_groups, group_size, imp in global_importance:
            module = group[0].dep.target.module
            pruning_fn = group[0].dep.handler
            get_channel_fn = self.DG.get_out_channels if self.DG.is_out_channel_pruning_fn(
                pruning_fn) else self.DG.get_in_channels

            # Prune feature dims/channels
            pruning_indices = []
            if len(global_importance) > 0 and n_pruned > 0:
                if ch_groups > 1:  # re-compute importance for each channel group if channel grouping is enabled
                    n_pruned_per_group = len((imp <= thres).nonzero().view(-1))
                    if n_pruned_per_group > 0:
                        if self.round_to:
                            n_pruned_per_group = self._round_to(n_pruned_per_group, group_size, self.round_to)
                        _is_attn, _ = self._is_attn_group(group)
                        if not _is_attn or self.prune_head_dims == True:
                            raw_imp = self.estimate_importance(group, ch_groups=ch_groups)  # re-compute importance
                            for chg in range(
                                    ch_groups):  # determine pruning indices for each channel group independently
                                sub_group_imp = raw_imp[chg * group_size: (chg + 1) * group_size]
                                sub_imp_argsort = torch.argsort(sub_group_imp)
                                sub_pruning_idxs = sub_imp_argsort[:n_pruned_per_group] + chg * group_size
                                pruning_indices.append(sub_pruning_idxs)
                else:
                    _pruning_indices = (imp <= thres).nonzero().view(-1)
                    imp_argsort = torch.argsort(imp)
                    if len(_pruning_indices) > 0 and self.round_to:
                        n_pruned = len(_pruning_indices)
                        current_channels = get_channel_fn(module)
                        n_pruned = self._round_to(n_pruned, current_channels, self.round_to)
                        _pruning_indices = imp_argsort[:n_pruned]
                    ### if all neurons are pruned, reserve one neuron
                    # if len(_pruning_indices) == len(imp):
                    #     _pruning_indices = _pruning_indices[:-1]
                    pruning_indices.append(_pruning_indices)


            # Prune heads
            if len(global_head_importance) > 0 and n_heads_removed > 0:
                if group in global_head_importance:
                    qkv_layers, head_imp = global_head_importance[group]
                    head_pruning_indices = (head_imp <= head_thres).nonzero().view(-1)
                    if len(head_pruning_indices) > 0:
                        for head_id in head_pruning_indices:
                            pruning_indices.append(
                                torch.arange(head_id * group_size, (head_id + 1) * group_size, device=head_imp.device))
                    for qkv_layer in qkv_layers:
                        self.num_heads[qkv_layer] -= len(head_pruning_indices)  # update num heads after pruning

            if len(pruning_indices) == 0: continue
            pruning_indices = torch.unique(torch.cat(pruning_indices, 0)).tolist()
            # create pruning group
            group = self.DG.get_pruning_group(
                module, pruning_fn, pruning_indices)
            if self.DG.check_pruning_group(group):
                yield group
