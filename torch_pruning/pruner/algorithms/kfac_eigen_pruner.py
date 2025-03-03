import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn as nn
import typing
from ... import ops
from .. import function
from ..kfac_utils.kfac_utils import (ComputeCovA,
                              ComputeCovG,
                              ComputeCovAPatch,
                              fetch_mat_weights,
                              mat_to_weight_and_bias)
from ..kfac_utils.common_utils import (tensor_to_list,
                                PresetLRScheduler)
from ..kfac_utils.prune_utils import (count_module_params,
                               get_rotation_layer_weights,
                               get_threshold,
                               filter_indices,
                               normalize_factors)
from tqdm import tqdm
from .kfac_pruner import KFACMetaPruner
from .scheduler import linear_scheduler

class KFACEigenPruner(KFACMetaPruner):

    def __init__(self,
                 model: nn.Module,  # a simple pytorch model
                 builder, # bottleneck builder
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
                 iterative_pruning_ratio_scheduler: typing.Callable = linear_scheduler,
                 # scheduler for iterative pruning.
                 ignored_layers: typing.List[nn.Module] = None,  # ignored layers
                 round_to: int = None,  # round channels to the nearest multiple of round_to

                 # Advanced
                 in_channel_groups: typing.Dict[nn.Module, int] = dict(),
                 # The number of channel groups for layer input
                 out_channel_groups: typing.Dict[nn.Module, int] = dict(),
                 # The number of channel groups for layer output
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
                 use_patch=True,
                 fix_layers=0):
        super(KFACEigenPruner, self).__init__(
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
            ch_sparsity_dict=ch_sparsity_dict,

            record_length=record_length,
            batch_averaged=batch_averaged,
            use_patch=use_patch,
            fix_layers=fix_layers
        )
        self.builder = builder
        self.CovAHandler = ComputeCovA() if not use_patch else ComputeCovAPatch()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        self.grad_outputs = {}
        self.model = model
        self.fix_layers = fix_layers
        # self._prepare_model()
        self.steps = 0
        self.use_patch = use_patch
        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.W_star = {}
        self.S_l = None

        self.importances = {}
        self._inversed = False

    def _save_input(self, module, input):
        aa = self.CovAHandler(input[0].data, module)
        # Initialize buffers
        if self.steps == 0:
            self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(0))
        self.m_aa[module] += aa

    def calculate_counts(self):
        counts = {}
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                origin = self.Q_a[module].numel() + self.Q_g[module].numel() + self.W_star[module].numel()
                counts[module] = origin
        return counts
    def step(self, interactive=False, re_init=False):
        self.current_step += 1
        pruning_method = self.prune_global if self.global_pruning else self.prune_local
        if interactive:  # yield groups for interactive pruning
            return pruning_method()
        else:
            self.remain_modules = []
            # origin_counts = self.calculate_counts()
            for group in pruning_method():
                ### update Q_a, Q_g, W_star
                for grp in group:
                    module = grp.dep.layer
                    if 'out' in str(grp.dep).split('=> ')[1].split(' ')[0]:
                        if isinstance(module, nn.Conv2d):
                            reserve_flag = torch.ones(module.weight.data.shape[0])
                        elif isinstance(module, nn.Linear):
                            reserve_flag = torch.ones(module.data.shape[0])
                        else:
                            continue
                        reserve_flag = reserve_flag.to(torch.bool)
                        reserve_flag[grp.idxs] = 0
                        # reserve_flag = reserve_flag.view(-1)
                        self.Q_g[module] = self.Q_g[module][:, reserve_flag]
                        self.W_star[module] = self.W_star[module][reserve_flag, :]
                    elif 'in' in str(grp.dep).split('=> ')[1].split(' ')[0]:
                        if isinstance(module, nn.Conv2d):
                            reserve_flag = torch.ones(module.weight.data.shape[1])
                        elif isinstance(module, nn.Linear):
                            reserve_flag = torch.ones(module.weight.data.shape[1])
                        else:
                            continue
                        reserve_flag = reserve_flag.to(torch.bool)
                        if module.bias is not None:
                            reserve_flag = torch.cat([reserve_flag, torch.ones(1).to(torch.bool)], dim=0)
                        reserve_flag[grp.idxs] = 0
                        self.Q_a[module] = self.Q_a[module][:, reserve_flag]
                        self.W_star[module] = self.W_star[module][..., reserve_flag]
                    else:
                        raise Exception("Unrecognized dependency type")
                group.prune()
            ### update remain_modules
            # pruned_counts = self.calculate_counts()
            # for module in self.model.modules():
            #     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            #         exact_ratio = pruned_counts[module] / origin_counts[module]
            #         if exact_ratio <= 1:
            #             self.remain_modules.append(module)
            self._rm_hooks()
            self._clear_buffer()
            # self._build_pruned_model(re_init)


    def _build_pruned_model(self, re_init):
        self.model = self.builder(self.model, True)
        self.model.register(self.remain_modules,
                            self.Q_g, self.Q_a,
                            self.W_star,
                            self.use_patch,
                            fix_rotation=False, re_init=re_init)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
        # Initialize buffers
        if self.steps == 0:
            self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(0))
        self.m_gg[module] += gg

    def _merge_Qs(self):
        for m, v in self.Q_g.items():
            if len(v) > 1:
                self.Q_g[m] = v[1] @ v[0]
            else:
                self.Q_g[m] = v[0]
        for m, v in self.Q_a.items():
            if len(v) > 1:
                self.Q_a[m] = v[1] @ v[0]
            else:
                self.Q_a[m] = v[0]

    def obtain_importance(self, dataloader, criterion, device, iter_steps=1000, fisher_type='true'):
        self._prepare_model()
        self.init_step()

        self._compute_fisher(dataloader, criterion, device, fisher_type, iter_steps)
        self._update_inv()  # eigen decomposition of fisher

        self._get_unit_importance()
        self._merge_Qs()  # update the eigen basis

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

    def _update_inv(self):
        assert self.steps > 0, 'At least one step before update inverse!'
        eps = 1e-10
        for idx, m in enumerate(self.modules):
            m_aa, m_gg = self.m_aa[m], self.m_gg[m]
            device = m_aa.device
            try:
                self.d_a[m], Q_a = torch.linalg.eigh(m_aa.to("cpu"))
            except:
                self.d_a[m], Q_a = torch.linalg.eigh(m_aa.to("cpu") + eps * torch.eye(m_aa.size(0)))
            try:
                self.d_g[m], Q_g = torch.linalg.eigh(m_gg.to("cpu"))
            except:
                self.d_g[m], Q_g = torch.linalg.eigh(m_gg.to("cpu") + eps * torch.eye(m_gg.size(0)))
            self.d_a[m], Q_a = self.d_a[m].to(device), Q_a.to(device)
            self.d_g[m], Q_g = self.d_g[m].to(device), Q_g.to(device)

            self.d_a[m].mul_((self.d_a[m] > eps).float())
            self.d_g[m].mul_((self.d_g[m] > eps).float())

            # == write summary ==
            name = m.__class__.__name__
            eigs = (self.d_g[m].view(-1, 1) @ self.d_a[m].view(1, -1)).view(-1).cpu().data.numpy()

            if self.Q_a.get(m, None) is None:
                # print('(%d)Q_a %s is None.' % (idx, m))
                self.Q_a[m] = [Q_a]  # absorb the eigen basis
            else:
                # self.Q_a[m] = [Q_a, self.Q_a[m]]
                ## from built model
                # prev_Q_a, prev_Q_g = get_rotation_layer_weights(self.model, m)
                ## from previous
                if isinstance(m, nn.Conv2d):
                    prev_Q_a = self.Q_a[m]
                    prev_Q_g = self.Q_g[m]
                elif isinstance(m, nn.Linear):
                    prev_Q_a = self.Q_a[m]
                    prev_Q_g = self.Q_g[m]
                else:
                    raise Exception("Unrecognized module type")

                prev_Q_a = prev_Q_a.view(prev_Q_a.size(0), prev_Q_a.size(1))
                prev_Q_g = prev_Q_g.view(prev_Q_g.size(0), prev_Q_g.size(1))
                self.Q_a[m] = [Q_a, prev_Q_a]

            if self.Q_g.get(m, None) is None:
                self.Q_g[m] = [Q_g]
            else:
                self.Q_g[m] = [Q_g, prev_Q_g]
        self._inversed = True

    def _get_unit_importance(self):
        assert self._inversed, 'Not inversed.'
        with torch.no_grad():
            for m in self.modules:
                w = fetch_mat_weights(m, self.use_patch)  # output_dim * input_dim
                # (Q_a ⊗ Q_g) vec(W) = Q_g.t() @ W @ Q_a
                if self.use_patch and isinstance(m, nn.Conv2d):
                    w_star_a = w.view(-1, w.size(-1)) @ self.Q_a[m][0]
                    w_star_g = self.Q_g[m][0].t() @ w_star_a.view(w.size(0), -1)
                    w_star = w_star_g.view(w.size())
                    if self.S_l is None:
                        w_imp = w_star ** 2 * (self.d_g[m].unsqueeze(1) @ self.d_a[m].unsqueeze(0)).unsqueeze(1)
                    else:
                        w_imp = w_star ** 2 * self.S_l[m]
                    # w_imp = w_imp.sum(dim=1)
                else:
                    w_star = self.Q_g[m][0].t() @ w @ self.Q_a[m][0]
                    if self.S_l is None:
                        w_imp = w_star ** 2 * (self.d_g[m].unsqueeze(1) @ self.d_a[m].unsqueeze(0))
                    else:
                        w_imp = w_star ** 2 * self.S_l[m]

                self.W_star[m] = w_star
                w_imp, b_imp = mat_to_weight_and_bias(w_imp, m)
                w_imp = w_imp.to('cpu')
                if m in self.importances.keys():
                    self.importances[m]['weight'] = torch.cat([self.importances[m]['weight'], w_imp.unsqueeze(0)],
                                                              dim=0)
                    if b_imp is not None:
                        self.importances[m]['bias'] = torch.cat([self.importances[m]['bias'], b_imp.unsqueeze(0)], dim=0)
                else:
                    self.importances[m] = {'weight': w_imp.unsqueeze(0)}
                    if b_imp is not None:
                        self.importances[m]['bias'] = b_imp.unsqueeze(0)
                if len(self.importances[m]['weight']) > self.record_length:
                    self.importances[m]['weight'] = self.importances[m]['weight'][1:]
                    if b_imp is not None:
                        self.importances[m]['bias'] = self.importances[m]['bias'][1:]

    def init_step(self):
        self.steps = 0

    def kfac_step(self):
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
        self.remain_modules = []
        self.modules = []
        self.W_star = {}
        if self.S_l is not None:
            self.S_l = {}

    def clear_all_buffer(self):
        self._clear_buffer()
        self.Q_a = {}
        self.Q_g = {}
        self.steps = 0
        self._inversed = False
        self.importances = {}