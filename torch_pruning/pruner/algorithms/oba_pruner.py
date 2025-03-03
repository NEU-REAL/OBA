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
import time
class OBAPruner(MetaPruner):
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
            delta: float = 1.0,
            upward_delta: float = 1.0,
            downward_delta: float = 1.0,
            parallel_delta: float = 1.0,
            other_unit_weight: bool = False,
            self_unit_weight: bool = False,
    ):
        super(OBAPruner, self).__init__(
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
        self.delta = delta
        self.upward_delta = upward_delta
        self.downward_delta = downward_delta
        self.parallel_delta = parallel_delta
        self.self_unit_weight = self_unit_weight
        self.other_unit_weight = other_unit_weight

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
        for module in self.model.modules():
            if module in self.group_importances.keys():
                weight_importance = self.group_importances[module]['weight'].abs() / self.group_importances[module]['count']
                ### You can manually change the code here to use different importance types
                # weight_importance = weight_importance ** 2
                # weight_importance = self.group_importances[module]['weight'] / self.group_importances[module]['count']
                # weight_importance = module.weight.data.abs()
                param_num += weight_importance.numel()
                importance = _normalize(weight_importance, self.importance.normalizer)
                importance += module.weight.data.abs() # resolve gradient vanishing
                separate_importance[module] = {}
                separate_importance[module]['weight'] = importance
                importance = importance.view(-1)
                if 'bias' in self.group_importances[module].keys():
                    bias_importance = self.group_importances[module]['bias'].abs()  / self.group_importances[module]['count']
                    ### You can manually change the code here to use different importance types
                    # bias_importance = bias_importance ** 2
                    # bias_importance = self.group_importances[module]['bias'] / self.group_importances[module]['count']
                    # bias_importance = module.bias.data.abs()
                    bias_importance = _normalize(bias_importance, self.importance.normalizer).view(-1)
                    bias_importance += module.bias.data.abs()  # resolve gradient vanishing
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
        for module in self.model.modules():
            if module in self.group_importances.keys():
                weight_importance = self.group_importances[module]['weight'].abs() / self.group_importances[module]['count']
                ### You can manually change the code here to use different importance types
                # weight_importance = weight_importance ** 2
                # weight_importance = self.group_importances[module]['weight'] / self.group_importances[module]['count']
                # weight_importance = module.weight.data.abs()
                param_num += weight_importance.numel()
                importance = _normalize(weight_importance, self.importance.normalizer)
                importance += module.weight.data.abs()
                separate_importance[module] = {}
                separate_importance[module]['weight'] = importance
                importance = importance.view(-1)
                if 'bias' in self.group_importances[module].keys():
                    bias_importance = self.group_importances[module]['bias'].abs()  / self.group_importances[module]['count']
                    ### You can manually change the code here to use different importance types
                    # bias_importance = bias_importance ** 2
                    # bias_importance = self.group_importances[module]['bias'] / self.group_importances[module]['count']
                    # bias_importance = module.bias.data.abs()
                    bias_importance = _normalize(bias_importance, self.importance.normalizer).view(-1)
                    bias_importance += module.bias.data.abs()
                    separate_importance[module]['bias'] = bias_importance
                    importance = torch.cat([importance, bias_importance])
                    param_num += bias_importance.numel()
                global_importance.append(importance)
        global_importance = torch.cat(global_importance)
        n_pruned = int(pruning_ratio * param_num)
        topk_imp, _ = torch.topk(global_importance, k=n_pruned, largest=True)
        # update parameters
        for module in self.model.modules():
            if module in self.group_importances.keys():
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

    def register_hooks(self):
        self.hook_handles = []
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention, Attention)):
                handle = module.register_forward_hook(forward_hook)
                self.hook_handles.append(handle)
                handle = module.register_full_backward_hook(backward_hook)
                self.hook_handles.append(handle)

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def obtain_importance(self, loss):
        current_group_importances = self.initialize_importance()
        current_group_importances = self.first_order_taylor(loss, current_group_importances)
        current_group_importances = self.both_connectivity_hessian(current_group_importances)
        self.update_group_importance(current_group_importances)

    def obtain_first_order_importance(self, loss):
        current_group_importances = self.initialize_importance()
        current_group_importances = self.first_order_taylor(loss, current_group_importances)
        self.update_group_importance(current_group_importances)

    def first_order_taylor(self, loss, current_group_importances):
        loss.backward(retain_graph=True)
        current_group_importances = self.record_gradient_x(current_group_importances)
        ## Obtain the first-order importance from upward layers
        for module in self.model.modules():
            if module in self.group_importances:
                if hasattr(module, "weight"):
                    if self.self_unit_weight:
                        delta_w = self.delta * torch.sign(module.weight.data)
                    else:
                        delta_w = self.delta * module.weight.data
                    if module.weight.grad is None:
                        dw = torch.zeros_like(module.weight.data)
                    else:
                        dw = module.weight.grad.data
                    importance = delta_w * dw
                    current_group_importances[module]["weight"] = importance.detach().clone()
                    current_group_importances[module]["first_order_weight"] = importance.detach().clone()
                if hasattr(module, "bias") and module.bias is not None:
                    b = module.bias.data
                    if self.self_unit_weight:
                        delta_b = self.delta * torch.sign(b)
                    else:
                        delta_b = self.delta * b
                    if module.weight.grad is None:
                        db = torch.zeros_like(b)
                    else:
                        db = module.bias.grad.data
                    importance = delta_b * db
                    current_group_importances[module]["bias"] = importance.detach().clone()
                    current_group_importances[module]["first_order_bias"] = importance.detach().clone()
        return current_group_importances

    def both_connectivity_hessian(self, current_group_importances):
        ## 1. Obtain the direct connectivity importance from upward layers
        current_group_importances = self.upward_direct_connectivity_hessian(current_group_importances)

        ## 2. Obtain the direct connectivity importance from downward layers
        current_group_importances = self.downward_both_connectivity_hessian(current_group_importances)
        return current_group_importances

    def upward_direct_connectivity_hessian(self, current_group_importances):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        for module in self.model.modules():
            if module in self.group_importances:
                if hasattr(module, "weight"):
                    if self.other_unit_weight:
                        delta_w = self.upward_delta * torch.sign(module.weight.data)
                    else:
                        delta_w = self.upward_delta * module.weight.data
                    x = module.X
                    if x.grad_fn is None:
                        continue
                    zero_bias = torch.zeros_like(module.bias.data) if hasattr(module, "bias") and module.bias is not None else None
                    y = self.surrogate_forward(module, x, delta_w, zero_bias)
                    y.backward(current_group_importances[module]["output_gradient"], retain_graph=True)
        for module in self.model.modules():
            if module in self.group_importances:
                if hasattr(module, "weight"):
                    dw = module.weight.grad.data
                    if self.self_unit_weight:
                        delta_w = self.upward_delta * torch.sign(module.weight.data)
                    else:
                        delta_w = self.upward_delta * module.weight.data
                    upward_hessian_importance = dw * delta_w
                    current_group_importances[module]["weight"] += upward_hessian_importance.detach().clone()
                if hasattr(module, "bias") and module.bias is not None:
                    db = module.bias.grad.data
                    if self.self_unit_weight:
                        delta_b = self.upward_delta * torch.sign(module.bias.data)
                    else:
                        delta_b = self.upward_delta * module.bias.data
                    upward_hessian_importance = db * delta_b
                    current_group_importances[module]["bias"] += upward_hessian_importance.detach().clone()
        return current_group_importances

    def check_zero_inputs(self, node, passed_nodes_feature):
        current_nodes = [node]
        while len(current_nodes) > 0:
            next_nodes = []
            for idx, current_node in enumerate(current_nodes):
                if len(current_node.inputs) > 0:
                    for next_node in current_node.inputs:
                        if next_node.module in passed_nodes_feature:
                            return False
                        if next_node not in next_nodes:
                            next_nodes.append(next_node)
            current_nodes = next_nodes
        return True

    def check_has_variable(self, node):
        current_nodes = [node]
        while len(current_nodes) > 0:
            next_nodes = []
            for idx, current_node in enumerate(current_nodes):
                if hasattr(current_node.grad_fn, "variable"):
                    return True
                if len(current_node.inputs) > 0:
                    for next_node in current_node.inputs:
                        if next_node not in next_nodes:
                            next_nodes.append(next_node)
            if len(next_nodes) == 0:
                current_fn = current_nodes[0].grad_fn
                if current_fn is not None:
                    while len(current_fn.next_functions) > 0:
                        next_fn = current_fn.next_functions[0][0]
                        if hasattr(next_fn, "variable"):
                            return True
                        if next_fn is None:
                            break
                        current_fn = next_fn
            current_nodes = next_nodes
        return False

    def obtain_feature(self, node, passed_nodes_feature):
        if node.module in passed_nodes_feature:
            return passed_nodes_feature[node.module]['hessian_feature']

        ## node version
        current_nodes = [node]
        past_nodes = []
        while len(current_nodes) > 0:
            next_nodes = []
            for idx, current_node in enumerate(current_nodes):
                if len(current_node.inputs) > 0:
                    for next_node in current_node.inputs:
                        if next_node not in next_nodes:
                            next_nodes.append(next_node)
                    past_nodes.append(current_node)
                elif hasattr(current_node.grad_fn, "variable"):
                    feature = current_node.grad_fn.variable.data
                    for past_node in reversed(past_nodes):
                        feature, _ = self.elementwise_surrogate_forward(past_node, feature, feature)
                    return feature
            if len(next_nodes) == 0:
                current_fn = current_nodes[0].grad_fn
                past_fns = []
                while len(current_fn.next_functions) > 0:
                    past_fns.append(current_fn)
                    next_fn = current_fn.next_functions[0][0]
                    if next_fn is None:
                        break
                    if hasattr(next_fn, "variable"):
                        feature = next_fn.variable.data
                        for past_fn in reversed(past_fns):
                            feature = self.elementwise_fn_surrogate_forward(past_fn, feature)
                        for past_node in reversed(past_nodes):
                            feature, _ = self.elementwise_surrogate_forward(past_node, feature, feature)
                        return feature
                    current_fn = next_fn
            current_nodes = next_nodes

    def downward_both_connectivity_hessian(self, current_group_importances):
        passed_nodes_feature = {}
        root_node = self.get_root_node(True)
        current_nodes = [root_node]
        is_root = True
        while len(current_nodes) > 0:
            unavail_nodes = []
            next_nodes = []
            for idx, current_node in enumerate(current_nodes):
                if all([node.module in passed_nodes_feature or self.check_zero_inputs(node, passed_nodes_feature) for node in
                        current_node.inputs]) or isinstance(current_node.module, nn.Linear) or isinstance(
                        current_node.module, nn.Conv2d):
                    passed_nodes_feature[current_node.module] = {}
                    module = current_node.module
                    valid_current_inputs = []
                    for node in current_node.inputs:
                        if self.check_has_variable(node) or node.module in passed_nodes_feature \
                                or isinstance(node.module, nn.Linear) or isinstance(node.module, nn.Conv2d):
                            valid_current_inputs.append(node)
                    input_hessian = tuple(self.obtain_feature(node, passed_nodes_feature) for node in
                                     valid_current_inputs) if not is_root else tuple()
                    input_forward = tuple(self.obtain_feature(node, passed_nodes_feature) for node in
                                     valid_current_inputs) if not is_root else tuple()
                    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                        if self.self_unit_weight:
                            delta_w = self.downward_delta * torch.sign(module.weight.data)
                            delta_bias = self.downward_delta * torch.sign(module.bias.data) if module.bias is not None else None
                        else:
                            delta_w = self.downward_delta * module.weight.data
                            delta_bias = self.downward_delta * module.bias.data if module.bias is not None else None
                        if len(input_hessian) > 0:## not root parameter module
                            input_hessian = (input_hessian[0].view(*module.X.shape),)
                            delta_w.requires_grad = True
                            output = self.surrogate_forward(module, *input_hessian, delta_w)
                            output_grad = current_group_importances[module]["output_gradient"].detach()
                            downward_to_current_hessian = torch.autograd.grad(output, delta_w, output_grad)[0] * delta_w
                            current_group_importances[module]["weight"] += downward_to_current_hessian.detach().clone()
                        # input_forward = (module.X.detach(),)
                        input_forward = (current_group_importances[module]["X"].detach(), )

                        if self.other_unit_weight:
                            delta_w = self.downward_delta * torch.sign(module.weight.data)
                            delta_bias = self.downward_delta * torch.sign(module.bias.data) if module.bias is not None else None
                        else:
                            delta_w = self.downward_delta * module.weight.data
                            delta_bias = self.downward_delta * module.bias.data if module.bias is not None else None
                        with torch.no_grad():
                            hessian_feature = self.surrogate_forward(module, *input_forward, delta_w, delta_bias)
                            if len(input_hessian) > 0:## not root parameter module
                                pre_hessian_feature = self.surrogate_forward(module, *input_hessian, module.weight.data,
                                                                             torch.zeros_like(
                                                                                 module.bias.data) if module.bias is not None else None)
                            else:
                                pre_hessian_feature = torch.zeros_like(hessian_feature)
                            forward_feature = self.surrogate_forward(module, *input_forward, module.weight.data)
                        passed_nodes_feature[module]['hessian_feature'] = hessian_feature + pre_hessian_feature
                        passed_nodes_feature[module]['forward_feature'] = forward_feature
                    elif 'BmmBackward' in current_node.name:
                        with torch.no_grad():
                            hessian_feature = torch.bmm(input_hessian[0], input_forward[1]) + torch.bmm(input_forward[0], input_hessian[1])
                            forward_feature = torch.bmm(input_forward[0], input_forward[1])
                        passed_nodes_feature[module]['left_forward_feature'] = input_forward[0]
                        passed_nodes_feature[module]['right_forward_feature'] = input_forward[1]
                        passed_nodes_feature[module]['left_hessian_feature'] = input_hessian[0]
                        passed_nodes_feature[module]['right_hessian_feature'] = input_hessian[1]
                        passed_nodes_feature[module]['hessian_feature'] = hessian_feature
                        passed_nodes_feature[module]['forward_feature'] = forward_feature
                    elif 'ScaledDotProductEfficientAttentionBackward0' in current_node.name:
                        passed_nodes_feature[module]['q_hessian'] = input_hessian[0].clone()
                        passed_nodes_feature[module]['k_hessian'] = input_hessian[1].clone()
                        passed_nodes_feature[module]['v_hessian'] = input_hessian[2].clone()
                        forward_q_grad = input_hessian[0].clone()
                        forward_k_grad = input_hessian[1].clone()
                        forward_v_grad = input_hessian[2].clone()

                        forward_q = input_forward[0]
                        forward_k = input_forward[1]
                        forward_v = input_forward[2]
                        with torch.no_grad():
                            forward_q_grad = forward_q_grad
                            forward_k_grad = forward_k_grad
                            forward_v_grad = forward_v_grad
                            forward_attn_weights = (forward_q @ forward_k.transpose(2, 3)) / torch.sqrt(
                                torch.tensor(forward_q.shape[-1]))
                            attn_weights_q = (forward_q_grad @ forward_k.transpose(2, 3)) / torch.sqrt(
                                torch.tensor(forward_q.shape[-1]))
                            attn_weights_k = (forward_q @ forward_k_grad.transpose(2, 3)) / torch.sqrt(
                                torch.tensor(forward_q.shape[-1]))
                            attn_weights_grad = attn_weights_q + attn_weights_k
                        softmax_forward_attn_weights = torch.softmax(forward_attn_weights, dim=-1)
                        jacobianfunc = torch.func.jacfwd(lambda x: torch.softmax(x, dim=0))
                        softmax_jacobian = torch.func.vmap(torch.func.vmap(torch.func.vmap(jacobianfunc)))(
                            forward_attn_weights).detach()
                        with torch.no_grad():
                            softmax_attn_weights_grad = (softmax_jacobian * attn_weights_grad.unsqueeze(3)).sum(4)
                            ## q, k gradient
                            out = softmax_attn_weights_grad @ forward_v + softmax_forward_attn_weights @ forward_v_grad
                        passed_nodes_feature[current_node.module]['hessian_feature'] = out
                    elif 'MultiheadAttention' in current_node.name:
                        forward_feature, hessian_feature = torch.func.jvp(
                            lambda x: current_node.module.forward(x.permute(1, 0, 2), x.permute(1, 0, 2), x.permute(1, 0, 2)),
                            input_forward, input_hessian)
                        passed_nodes_feature[module]['q_hessian'] = input_hessian[0].clone()
                        passed_nodes_feature[module]['k_hessian'] = input_hessian[0].clone()
                        passed_nodes_feature[module]['v_hessian'] = input_hessian[0].clone()
                        passed_nodes_feature[module]['hessian_feature'] = hessian_feature
                        passed_nodes_feature[module]['forward_feature'] = forward_feature
                    else:
                        with torch.no_grad():
                            hessian_feature, forward_feature = self.elementwise_surrogate_forward(current_node,
                                                                                                  input_hessian,
                                                                                                  input_forward)
                        passed_nodes_feature[module]['hessian_feature'] = hessian_feature
                        passed_nodes_feature[module]['forward_feature'] = forward_feature
                    for node in current_node.outputs:
                        if node not in next_nodes and node.module not in passed_nodes_feature:
                            next_nodes.append(node)
                else:
                    if current_node not in unavail_nodes:
                        unavail_nodes.append(current_node)
            next_nodes.extend(unavail_nodes)
            is_root = False
            current_nodes = next_nodes
        current_group_importances = self.obtain_parallel_importance(passed_nodes_feature, current_group_importances)
        return current_group_importances

    def obtain_parallel_importance(self, passed_nodes_feature, current_group_importances):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        for module in self.model.modules():
            if isinstance(module, Attention) or isinstance(module, nn.MultiheadAttention):
                # q, k, v = module.X
                q, k, v = current_group_importances[module]['X']
                b, n, emb_dim = q.shape
                try:
                    out_proj_node = self.DG.module2node[module.out_proj]
                    bmm_mode = True
                except:
                    bmm_mode = False
                if bmm_mode:
                    bmm_node_list = []
                    current_nodes = [out_proj_node]
                    while len(bmm_node_list) < 2:
                        next_nodes = []
                        for current_node in current_nodes:
                            for node in current_node.inputs:
                                if node not in next_nodes:
                                    next_nodes.append(node)
                        for next_node in next_nodes:
                            if 'BmmBackward' in next_node.name:
                                bmm_node_list.append(next_node)
                        current_nodes = next_nodes
                    forward_v_grad = passed_nodes_feature[bmm_node_list[0].module]['right_hessian_feature']
                    forward_q_grad = passed_nodes_feature[bmm_node_list[1].module]['left_hessian_feature']
                    forward_k_grad = passed_nodes_feature[bmm_node_list[1].module]['right_hessian_feature'].transpose(1, 2).contiguous()
                    forward_q = q.reshape(b, n, module.num_heads, module.head_dim).permute(0, 2, 1, 3)
                    forward_k = k.reshape(b, n, module.num_heads, module.head_dim).permute(0, 2, 1, 3)
                    forward_v = v.reshape(b, n, module.num_heads, module.head_dim).permute(0, 2, 1, 3)
                    forward_q_grad = forward_q_grad.reshape(b, n, module.num_heads, module.head_dim).permute(0, 2, 1, 3)
                    forward_k_grad = forward_k_grad.reshape(b, n, module.num_heads, module.head_dim).permute(0, 2, 1, 3)
                    forward_v_grad = forward_v_grad.reshape(b, n, module.num_heads, module.head_dim).permute(0, 2, 1, 3)
                else:
                    forward_v_grad = passed_nodes_feature[module]['v_hessian'].transpose(0, 1).contiguous()
                    forward_q_grad = passed_nodes_feature[module]['q_hessian'].transpose(0, 1).contiguous()
                    forward_k_grad = passed_nodes_feature[module]['k_hessian'].transpose(0, 1).contiguous()
                    forward_q = q.reshape(b, n, module.num_heads, module.head_dim).permute(0, 2, 1, 3)
                    forward_k = k.reshape(b, n, module.num_heads, module.head_dim).permute(0, 2, 1, 3)
                    forward_v = v.reshape(b, n, module.num_heads, module.head_dim).permute(0, 2, 1, 3)
                    forward_q_grad = forward_q_grad.reshape(b, n, module.num_heads, module.head_dim).permute(0, 2, 1, 3)
                    forward_k_grad = forward_k_grad.reshape(b, n, module.num_heads, module.head_dim).permute(0, 2, 1, 3)
                    forward_v_grad = forward_v_grad.reshape(b, n, module.num_heads, module.head_dim).permute(0, 2, 1, 3)

                current_group_importances[module]["forward_q_grad"] = forward_q_grad
                current_group_importances[module]["forward_k_grad"] = forward_k_grad
                current_group_importances[module]["forward_v_grad"] = forward_v_grad
                forward_attn_weights = (forward_q @ forward_k.transpose(2, 3)) / torch.sqrt(
                    torch.tensor(module.head_dim))
                attn_weights_q = (forward_q_grad @ forward_k.transpose(2, 3)) / torch.sqrt(
                    torch.tensor(module.head_dim))
                attn_weights_k = (forward_q @ forward_k_grad.transpose(2, 3)) / torch.sqrt(
                    torch.tensor(module.head_dim))
                attn_weights_grad = attn_weights_q + attn_weights_k
                softmax_forward_attn_weights = torch.softmax(forward_attn_weights, dim=-1)
                jacobianfunc = torch.func.jacfwd(lambda x: torch.softmax(x, dim=0))
                softmax_jacobian = torch.func.vmap(torch.func.vmap(torch.func.vmap(jacobianfunc)))(
                    forward_attn_weights).detach()
                softmax_attn_weights_grad = (softmax_jacobian * attn_weights_grad.unsqueeze(3)).sum(4)
                before_linear_grad = (current_group_importances[module]['output_gradient'].unsqueeze(
                    -1) * module.out_proj.weight).sum(2).detach()
                ## q, k gradient
                out = softmax_attn_weights_grad @ forward_v.detach()
                out = out.reshape(b, module.num_heads, n, module.head_dim).permute(0, 2, 1, 3).reshape(b, n,
                                                                                                       module.num_heads * module.head_dim)
                out.backward(before_linear_grad, retain_graph=True)
                # ## v gradient
                out = softmax_forward_attn_weights @ forward_v_grad
                out = out.reshape(b, module.num_heads, n, module.head_dim).permute(0, 2, 1, 3).reshape(b, n,
                                                                                                       module.num_heads * module.head_dim)
                out.backward(before_linear_grad, retain_graph=True)
                ## attn weights gradient
                out = softmax_attn_weights_grad.detach() @ forward_v
                out = out.reshape(b, module.num_heads, n, module.head_dim).permute(0, 2, 1, 3).reshape(b, n,
                                                                                                       module.num_heads * module.head_dim)
                out.backward(before_linear_grad, retain_graph=True)
        for module in self.model.modules():
            if module in self.group_importances:
                if hasattr(module, "weight"):
                    dw = module.weight.grad.data
                    if self.self_unit_weight:
                        delta_w = self.parallel_delta * torch.sign(module.weight.data)
                    else:
                        delta_w = self.parallel_delta * module.weight.data
                    parallel_hessian_importance = dw * delta_w
                    current_group_importances[module]["weight"] += parallel_hessian_importance.detach()
                if hasattr(module, "bias") and module.bias is not None:
                    db = module.bias.grad.data
                    if self.self_unit_weight:
                        delta_b = self.parallel_delta * torch.sign(module.bias.data)
                    else:
                        delta_b = self.parallel_delta * module.bias.data
                    parallel_hessian_importance = db * delta_b
                    current_group_importances[module]["bias"] += parallel_hessian_importance.detach()
        return current_group_importances

    def get_root_node(self, parameter_node=False):
        root_node = list(self.DG.module2node.items())[0][1]
        current_nodes = [root_node]
        has_input = True
        while has_input:
            next_nodes = []
            has_input = False
            for cur_node in current_nodes:
                if len(cur_node.inputs) > 0:
                    has_input = True
                    for sub_node in cur_node.inputs:
                        if sub_node not in next_nodes:
                            next_nodes.append(sub_node)
            if len(next_nodes) > 0:
                current_nodes = next_nodes
        root_node = current_nodes[0]
        if parameter_node:
            while not isinstance(root_node.module, nn.Linear) and not isinstance(root_node.module, nn.Conv2d):
                root_node = root_node.outputs[0]
        return root_node


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
                    else:
                        raise Exception("Unrecognized layer weight shape")
                    layer_bias = current_group_importances[module]['bias'] if 'bias' in current_group_importances[module].keys() else None
                    if layer_bias is not None:
                        output_imp += layer_bias
                    self.group_importances[module]["input"] = torch.cat(
                        [self.group_importances[module]["input"], input_imp.unsqueeze(0).cpu()])
                    if self.group_importances[module]["input"].shape[0] > self.record_length:
                        self.group_importances[module]["input"] = self.group_importances[module]["input"][1:]
                    self.group_importances[module]["output"] = torch.cat(
                        [self.group_importances[module]["output"], output_imp.unsqueeze(0).cpu()])
                    if self.group_importances[module]["output"].shape[0] > self.record_length:
                        self.group_importances[module]["output"] = self.group_importances[module]["output"][1:]


                    # update weight importance
                    self.group_importances[module]["weight"] += layer_weight.to(torch.float64)
                    if layer_bias is not None:
                        self.group_importances[module]["bias"] += layer_bias.to(torch.float64)
                    self.group_importances[module]["count"] += 1


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

    def surrogate_forward(self, module, x, weight, bias=None):
        if isinstance(module, nn.Linear):
            return torch.nn.functional.linear(x, weight, bias if bias is not None else module.bias)
        elif isinstance(module, nn.Conv2d):
            return torch.nn.functional.conv2d(x, weight, bias if bias is not None else module.bias,
                                              module.stride, module.padding, module.dilation, module.groups)
        else:
            raise NotImplementedError

    def elementwise_fn_surrogate_forward(self, grad_fn, feature):
        if  "TBackward" in str(grad_fn):
            return feature.T
        else:
            raise NotImplementedError

    def elementwise_surrogate_forward(self, node, hessian_input, forward_input):
        if isinstance(node.module, nn.BatchNorm2d):
            with torch.no_grad():
                output, jvp_output = torch.func.jvp(
                    lambda x: nn.functional.batch_norm(x, node.module.running_mean, node.module.running_var,
                                                       node.module.weight,
                                                       node.module.bias, node.module.training, node.module.momentum,
                                                       node.module.eps), forward_input, hessian_input)
            return jvp_output, output
        elif isinstance(node.module, nn.LayerNorm):
            with torch.no_grad():
                output, jvp_output = torch.func.jvp(
                    lambda x: nn.functional.layer_norm(x, node.module.normalized_shape, node.module.weight,
                                                       node.module.bias, node.module.eps), forward_input, hessian_input)
            return jvp_output, output
        elif 'ReluBackward' in node.name:
            grad_mask = (forward_input[0] > 0).float()
            return hessian_input[0] * grad_mask, forward_input[0] * grad_mask
        elif 'AddmmBackward' in node.name:
            return hessian_input[0], forward_input[0]
        elif 'AddBackward' in node.name:
            if len(hessian_input) == 1:
                return hessian_input[0], forward_input[0]
            else:
                if isinstance(hessian_input[0], torch.Tensor):
                    hessian_input_0 = hessian_input[0]
                    forward_input_0 = forward_input[0]
                else:
                    hessian_input_0 = hessian_input[0][0]
                    forward_input_0 = forward_input[0][0]
                if isinstance(hessian_input[1], torch.Tensor):
                    hessian_input_1 = hessian_input[1]
                    forward_input_1 = forward_input[1]
                else:
                    hessian_input_1 = hessian_input[1][0]
                    forward_input_1 = forward_input[1][0]
                return hessian_input_0 + hessian_input_1, forward_input_0 + forward_input_1
        elif 'AvgPool2DBackward' in node.name and not 'Adaptive' in node.name:
            stride = node.grad_fn._saved_stride[0] if len(node.grad_fn._saved_stride) > 0 else None
            return nn.functional.avg_pool2d(hessian_input[0], kernel_size=node.grad_fn._saved_kernel_size[0],
                                            stride=stride,
                                            padding=node.grad_fn._saved_padding[0]), nn.functional.avg_pool2d(
                forward_input[0], kernel_size=node.grad_fn._saved_kernel_size[0],
                stride=stride, padding=node.grad_fn._saved_padding[0])
        elif 'AdaptiveAvgPool2DBackward' in node.name:
            if not hasattr(node, '_saved_output_size'):
                post_node = node.outputs[0]
                post_nodes = [post_node]
                while not (isinstance(post_node.module, nn.Linear) or isinstance(post_node.module, nn.Conv2d)):
                    post_node = post_node.outputs[0]
                    post_nodes.append(post_node)
                feature = post_node.module.Y.detach()
                for idx, current_node in reversed(list(enumerate(post_nodes))):
                    if idx > 0:
                        pre_node = post_nodes[idx - 1]
                    else:
                        pre_node = node
                    node_input_idx = torch.tensor([pre_node.grad_fn is temp_gradfn[0] for temp_gradfn in current_node.grad_fn.next_functions]).nonzero().item()
                    if hasattr(current_node.grad_fn, '__call__'):
                        feature = current_node.grad_fn(feature)
                        if isinstance(feature, tuple):
                            feature = feature[node_input_idx]
                node._saved_output_size = feature.shape[-2:]
            return nn.functional.adaptive_avg_pool2d(hessian_input[0], node._saved_output_size), nn.functional.adaptive_avg_pool2d(
                forward_input[0], node._saved_output_size)
        elif 'Reshape' in node.name:
            if not hasattr(node, '_saved_output_size'):
                post_node = node.outputs[0]
                post_nodes = [post_node]
                while not (isinstance(post_node.module, nn.Linear) or isinstance(post_node.module, nn.Conv2d)):
                    post_node = post_node.outputs[0]
                    post_nodes.append(post_node)
                feature = post_node.module.Y.detach()
                for idx, current_node in reversed(list(enumerate(post_nodes))):
                    if idx > 0:
                        pre_node = post_nodes[idx - 1]
                    else:
                        pre_node = node
                    node_input_idx = torch.tensor([pre_node.grad_fn is temp_gradfn[0] for temp_gradfn in current_node.grad_fn.next_functions]).nonzero().item()
                    if hasattr(current_node.grad_fn, '__call__'):
                        feature = current_node.grad_fn(feature)
                        if isinstance(feature, tuple):
                            feature = feature[node_input_idx]
                node._saved_output_size = feature.shape
            return hessian_input[0].reshape(*node._saved_output_size), forward_input[0].reshape(*node._saved_output_size)
        elif 'HookFunction' in node.name:
            return hessian_input[0], forward_input[0]
        elif 'SiluBackward' in node.name:
            output, jvp_output = torch.func.jvp(torch.nn.functional.silu, forward_input, hessian_input)
            return jvp_output, output
        elif 'MmBackward' in node.name:
            if len(hessian_input) == 1:
                return hessian_input[0], forward_input[0]
            else:
                return hessian_input[0].mm(hessian_input[1]), forward_input[0].mm(forward_input[1])
        elif 'PermuteBackward' in node.name:
            return hessian_input[0].permute(*node.grad_fn._saved_dims), forward_input[0].permute(*node.grad_fn._saved_dims)
        elif 'SelectBackward' in node.name:
            return hessian_input[0].select(node.grad_fn._saved_dim, node.grad_fn._saved_index), forward_input[0].select(
                node.grad_fn._saved_dim, node.grad_fn._saved_index)
        elif 'ExpandBackward' in node.name:
            if not hasattr(node, '_saved_output_size'):
                post_node = node.outputs[0]
                post_nodes = [post_node]
                while not (isinstance(post_node.module, nn.Linear) or isinstance(post_node.module, nn.Conv2d)):
                    post_node = post_node.outputs[0]
                    post_nodes.append(post_node)
                feature = post_node.module.Y.detach()
                for idx, current_node in reversed(list(enumerate(post_nodes))):
                    if idx > 0:
                        pre_node = post_nodes[idx - 1]
                    else:
                        pre_node = node
                    node_input_idx = torch.tensor([pre_node.grad_fn is temp_gradfn[0] for temp_gradfn in current_node.grad_fn.next_functions]).nonzero().item()
                    feature = current_node.grad_fn(feature)
                    if isinstance(feature, tuple):
                        feature = feature[node_input_idx]
                node._saved_output_size = feature.shape
            return hessian_input[0].expand(*node._saved_output_size), forward_input[0].expand(*node._saved_output_size)
        elif 'TransposeBackward' in node.name:
            dim0 = node.grad_fn._saved_dim0 - 18446744073709551616 if node.grad_fn._saved_dim0 > 18000000000000000000 else node.grad_fn._saved_dim0
            dim1 = node.grad_fn._saved_dim1 - 18446744073709551616 if node.grad_fn._saved_dim1 > 18000000000000000000 else node.grad_fn._saved_dim1
            return hessian_input[0].transpose(dim0, dim1), forward_input[0].transpose(dim0, dim1)
        elif 'CloneBackward' in node.name:
            return hessian_input[0], forward_input[0]
        elif 'DivBackward' in node.name:
            return hessian_input[0] / node.grad_fn._saved_other, forward_input[0] / node.grad_fn._saved_other
        elif 'LogSoftmaxBackward' in node.name:
            if not hasattr(node, '_logsoftmax_dim'):
                for dim in range(len(node.grad_fn._saved_result.shape)):
                    if torch.allclose(node.grad_fn._saved_result.exp().sum(dim), torch.ones_like(node.grad_fn._saved_result.sum(dim))):
                        node._logsoftmax_dim = dim
                        break
            output, jvp_output = torch.func.jvp(lambda x: torch.log_softmax(x, dim=node._logsoftmax_dim), forward_input, hessian_input)
            return jvp_output, output
        elif 'SoftmaxBackward' in node.name:
            if not hasattr(node, '_softmax_dim'):
                for dim in range(len(node.grad_fn._saved_result.shape)):
                    if torch.allclose(node.grad_fn._saved_result.sum(dim), torch.ones_like(node.grad_fn._saved_result.sum(dim))):
                        node._softmax_dim = dim
                        break
            output, jvp_output = torch.func.jvp(lambda x: torch.softmax(x, dim=node._softmax_dim), forward_input, hessian_input)
            return jvp_output, output
        elif 'MaxPool2DWithIndicesBackward' in node.name:
            output, jvp_output = torch.func.jvp(lambda x: nn.functional.max_pool2d(x, kernel_size=node.grad_fn._saved_kernel_size[0],
                                            stride=node.grad_fn._saved_stride[0],
                                            padding=node.grad_fn._saved_padding[0]), forward_input, hessian_input)
            return jvp_output, output
        elif 'MeanBackward1' in node.name:
            dim = [a - 18446744073709551616 if a > 18000000000000000000 else a for a in node.grad_fn._saved_dim]
            hessian = hessian_input[0].mean(dim, keepdim=node.grad_fn._saved_keepdim)
            forward = forward_input[0].mean(dim, keepdim=node.grad_fn._saved_keepdim)
            return hessian, forward
        elif 'ConcatOp' in node.name:
            return torch.cat(hessian_input, dim=node.grad_fn._saved_dim), torch.cat(forward_input, dim=node.grad_fn._saved_dim)
        elif 'UnsqueezeBackward' in node.name:
            return hessian_input[0].unsqueeze(node.grad_fn._saved_dim), forward_input[0].unsqueeze(node.grad_fn._saved_dim)
        elif 'SqueezeBackward' in node.name:
            dim = node.grad_fn._saved_dim - 18446744073709551616 if node.grad_fn._saved_dim > 18000000000000000000 else node.grad_fn._saved_dim
            return hessian_input[0].squeeze(dim), forward_input[0].squeeze(dim)
        elif 'GeluBackward' in node.name:
            output, jvp_output = torch.func.jvp(torch.nn.functional.gelu, forward_input, hessian_input)
            return jvp_output, output
        elif 'SliceBackward' in node.name:
            start_idx = node.grad_fn._saved_start - 18446744073709551616 if node.grad_fn._saved_start > 18000000000000000000 else node.grad_fn._saved_start
            end_idx = node.grad_fn._saved_end - 9223372036854775808 if node.grad_fn._saved_end > 9000000000000000000 else node.grad_fn._saved_end
            end_idx += 1
            if end_idx == 0:
                end_idx = hessian_input[0].shape[node.grad_fn._saved_dim]
            step = node.grad_fn._saved_step
            if node.grad_fn._saved_dim == 0:
                return hessian_input[0][start_idx:end_idx:step], forward_input[0][start_idx:end_idx:step]
            elif node.grad_fn._saved_dim == 1:
                return hessian_input[0][:, start_idx:end_idx:step], forward_input[0][:, start_idx:end_idx:step]
            elif node.grad_fn._saved_dim == 2:
                return hessian_input[0][:, :, start_idx:end_idx:step], forward_input[0][:, :, start_idx:end_idx:step]
            elif node.grad_fn._saved_dim == 3:
                return hessian_input[0][:, :, :, start_idx:end_idx:step], forward_input[0][:, :, :, start_idx:end_idx:step]
            elif node.grad_fn._saved_dim == 4:
                return hessian_input[0][:, :, :, :, start_idx:end_idx:step], forward_input[0][:, :, :, :, start_idx:end_idx:step]
            else:
                raise Exception("Undefined, Please define it above")
        else:
            raise NotImplementedError('The jvp forward of current layer type is not implemented yet.')



    def initialize_importance(self):
        group_importances = {}
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, torch.nn.MultiheadAttention, Attention)) and not isinstance(module, torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
                group_importances[module] = {}
                if hasattr(module, "weight"):
                    group_importances[module]["weight"] = torch.zeros_like(module.weight.data).to(torch.float64)
                # group_importances[module]["first_order_weight"] = torch.tensor([])
                if hasattr(module, "bias") and module.bias is not None:
                    group_importances[module]["bias"] = torch.zeros_like(module.bias.data).to(torch.float64)
                    # group_importances[module]["first_order_bias"] = torch.tensor([])
                group_importances[module]["input"] = torch.tensor([])
                group_importances[module]["output"] = torch.tensor([])
                group_importances[module]["count"] = 0
        return group_importances

    def record_gradient_x(self, group_importances):
        for module in self.model.modules():
            if module in self.group_importances:
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention, Attention)):
                    group_importances[module]["output_gradient"] = module.grad_output[0].detach().clone()
                    group_importances[module]["X"] = module.X
        return group_importances


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
