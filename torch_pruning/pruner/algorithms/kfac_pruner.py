import torch
import torch.nn as nn
import typing, warnings
from .metapruner import MetaPruner
from .. import function
from .scheduler import linear_scheduler
from ... import ops
from collections import OrderedDict
from ..kfac_utils.kfac_utils import (ComputeCovA,
                              ComputeCovAPatch,
                              ComputeCovG,
                              fetch_mat_weights,
                              mat_to_weight_and_bias)
from ..kfac_utils.common_utils import (tensor_to_list, PresetLRScheduler)
from ..kfac_utils.prune_utils import (filter_indices,
                               get_threshold,
                               update_indices,
                               normalize_factors)

class KFACMetaPruner(MetaPruner):
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
            iterative_pruning_ratio_scheduler: typing.Callable = linear_scheduler,  # scheduler for iterative pruning.
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
            record_length: int = 500,
            batch_averaged=True,
            use_patch=False,
            fix_layers=0
    ):
        super(KFACMetaPruner, self).__init__(
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
        self.importances = {}
        self.record_length = record_length
        self.batch_averaged = batch_averaged
        self.use_patch = use_patch
        self.modules = []
        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.CovAHandler = ComputeCovA() if not use_patch else ComputeCovAPatch()
        self.CovGHandler = ComputeCovG()
        self.W_pruned = {}
        self.known_modules = {'Linear', 'Conv2d'}
        self.S_l = None
        self.fix_layers = fix_layers

    def estimate_importance(self, group, ch_groups=1) -> torch.Tensor:
        return self.importance(group, self.importances, ch_groups=ch_groups)

    def step(self, interactive=False):
        self.current_step += 1
        pruning_method = self.prune_global if self.global_pruning else self.prune_local

        if interactive:  # yield groups for interactive pruning
            return pruning_method()
        else:
            for group in pruning_method():
                group.prune()
                # adjust group importances for one pass pruning
                for grp in group:
                    if grp.dep.layer in self.importances.keys():
                        module = grp.dep.layer
                        if 'out' in str(grp).split('=> ')[1].split(' ')[0]:
                            is_out = True
                            reserve_idxs = torch.ones(self.importances[module]['output'].shape[1]).bool()
                        elif 'in' in str(grp).split('=> ')[1].split(' ')[0]:
                            is_out = False
                            reserve_idxs = torch.ones(self.importances[module]['input'].shape[1]).bool()
                        else:
                            raise Exception("Unrecognized dimension")
                        prune_idxs = grp.idxs
                        module = grp.dep.layer
                        reserve_idxs[prune_idxs] = False
                        if is_out:
                            self.importances[module]['output'] = self.importances[module]['output'][:,
                                                                       reserve_idxs]
                        else:
                            self.importances[module]['input'] = self.importances[module]['input'][:,
                                                                      reserve_idxs]

    def _save_input(self, module, input):
        aa = self.CovAHandler(input[0].data, module)
        # Initialize buffers
        if self.steps == 0:
            self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(0))
        self.m_aa[module] += aa

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
        # Initialize buffers
        if self.steps == 0:
            self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(0))
        self.m_gg[module] += gg

    def obtain_importance(self, dataloader, criterion, device, iter_steps=1000, fisher_type='true'):
        self._prepare_model()
        self.init_step()

        self._compute_fisher(dataloader, criterion, device, fisher_type, iter_steps=iter_steps)
        self._update_inv()  # eigen decomposition of fisher

        self._get_unit_importance()
        self._rm_hooks()
        self._clear_buffer()

    def _prepare_model(self):
        count = 0
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                count += 1
        self.modules = self.modules[self.fix_layers:]

    def _compute_fisher(self, dataloader, criterion, device='cuda', fisher_type='true', iter_steps=1000):
        self.mode = 'basis'
        self.model = self.model.eval()
        self.init_step()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.model(inputs)
            if fisher_type == 'true':
                sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                              1).squeeze().to(device)
                loss_sample = criterion(outputs, sampled_y)
                loss_sample.backward()
            else:
                loss = criterion(outputs, targets)
                loss.backward()
            self.kfac_step()
            if self.steps >= iter_steps:
                break
        self.mode = 'quite'

    def _update_inv(self):
        assert self.steps > 0, 'At least one step before update inverse!'
        eps = 1e-15
        for idx, m in enumerate(self.modules):
            # m_aa, m_gg = normalize_factors(self.m_aa[m], self.m_gg[m])
            m_aa, m_gg = self.m_aa[m], self.m_gg[m]
            device = m_aa.device
            try:
                self.d_a[m], self.Q_a[m] = torch.linalg.eigh(m_aa.to("cpu") / self.steps)
            except:
                self.d_a[m], self.Q_a[m] = torch.linalg.eigh(m_aa.to("cpu") / self.steps + eps * torch.eye(m_aa.size(0)))
            try:
                self.d_g[m], self.Q_g[m] = torch.linalg.eigh(m_gg.to("cpu") / self.steps)
            except:
                self.d_g[m], self.Q_g[m] = torch.linalg.eigh(m_gg.to("cpu") / self.steps + eps * torch.eye(m_aa.size(0)))
            self.d_a[m], self.Q_a[m] = self.d_a[m].to(device), self.Q_a[m].to(device)
            self.d_g[m], self.Q_g[m] = self.d_g[m].to(device), self.Q_g[m].to(device)
            self.d_a[m].mul_((self.d_a[m] > eps).float())
            self.d_g[m].mul_((self.d_g[m] > eps).float())

        self._inversed = True

    def _get_unit_importance(self):
        eps = 1e-10
        assert self._inversed, 'Not inversed.'
        with torch.no_grad():
            for m in self.modules:
                w = fetch_mat_weights(m, False)  # output_dim * input_dim
                # (Q_a ⊗ Q_g) vec(W) = Q_g.t() @ W @ Q_a
                if self.S_l is None:
                    A_inv = self.Q_a[m] @ (torch.diag(1.0 / (self.d_a[m] + eps))) @ self.Q_a[m].t()
                    G_inv = self.Q_g[m] @ (torch.diag(1.0 / (self.d_g[m] + eps))) @ self.Q_g[m].t()
                    A_inv_diag = torch.diag(A_inv)
                    G_inv_diag = torch.diag(G_inv)
                    w_imp = w ** 2 / (G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0))
                else:
                    Q_a, Q_g = self.Q_a[m], self.Q_g[m]
                    S_l = self.S_l[m]
                    S_l_inv = 1.0 / (S_l + eps)
                    H_inv_diag = (Q_g ** 2) @ S_l_inv @ (Q_a.t() ** 2)  # output_dim * input_dim
                    w_imp = w ** 2 / H_inv_diag
                self.W_pruned[m] = w
                w_imp, b_imp = mat_to_weight_and_bias(w_imp, m)
                w_imp = w_imp.to('cpu')
                layer_weight = w_imp
                if len(layer_weight.shape) == 4:
                    input_imp = layer_weight.sum((0, 2, 3))
                    output_imp = layer_weight.sum((1, 2, 3))
                elif len(layer_weight.shape) == 2:
                    input_imp = layer_weight.sum(0)
                    output_imp = layer_weight.sum(1)
                else:
                    raise Exception('Unknown layer shape')
                layer_bias = b_imp
                if layer_bias is not None:
                    output_imp += layer_bias.cpu()
                if m in self.importances.keys():
                    self.importances[m]["input"] = torch.cat(
                        [self.importances[m]["input"], input_imp.unsqueeze(0).cpu()])
                    if self.importances[m]["input"].shape[0] > self.record_length:
                        self.importances[m]["input"] = self.importances[m]["input"][1:]
                    self.importances[m]["output"] = torch.cat(
                        [self.importances[m]["output"], output_imp.unsqueeze(0).cpu()])
                    if self.importances[m]["output"].shape[0] > self.record_length:
                        self.importances[m]["output"] = self.importances[m]["output"][1:]
                else:
                    self.importances[m] = {'output': output_imp.unsqueeze(0).cpu()}
                    self.importances[m]['input'] = input_imp.unsqueeze(0).cpu()

    def init_step(self):
        self.steps = 0

    def kfac_step(self): # raw step method in KFAC is renamed to kfac_step
        self.steps += 1

    def _rm_hooks(self):
        for m in self.model.modules():
            classname = m.__class__.__name__
            if classname in self.known_modules:
                m._backward_hooks = OrderedDict()
                m._forward_pre_hooks = OrderedDict()

    def _clear_buffer(self):
        self.m_aa = {}
        self.m_gg = {}
        self.d_a = {}
        self.d_g = {}
        self.Q_a = {}
        self.Q_g = {}
        self.modules = []
        if self.S_l is not None:
            self.S_l = {}