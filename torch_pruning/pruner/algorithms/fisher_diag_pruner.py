import torch
import torch.nn as nn
import typing
from ... import ops
from .. import function
from .kfac_pruner import KFACMetaPruner
from ..kfac_utils.common_utils import tensor_to_list
from ..kfac_utils.kfac_utils import (ComputeMatGrad,
                              fetch_mat_weights,
                              mat_to_weight_and_bias)
from .scheduler import linear_scheduler

class FisherDiagPruner(KFACMetaPruner):

    def __init__(self,
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
                 use_patch=False,
                 fix_layers=0):

        super(FisherDiagPruner, self).__init__(
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
        print('Using patch is %s' % use_patch)
        self.use_patch = False
        self.MatGradHandler = ComputeMatGrad()

        self.batch_averaged = batch_averaged
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        self.model = model
        self.fix_layers = fix_layers
        self.steps = 0
        self.W_pruned = {}

        self.importances = {}
        self._inversed = False
        self._cfgs = {}
        self._indices = {}

        self.A, self.DS = {}, {}
        self.Fisher = {}

    def _save_input(self, module, input):
        self.A[module] = input[0].data

    def _save_grad_output(self, module, grad_input, grad_output):
        self.DS[module] = grad_output[0].data

    def obtain_importance(self, dataloader, criterion, device, iter_steps=1000, fisher_type='true'):
        self._prepare_model()
        self.init_step()

        temp_loader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=64, shuffle=True,
                                                  num_workers=2)
        self._compute_fisher(temp_loader, criterion, device, fisher_type, iter_steps)

        self._get_unit_importance()
        self._rm_hooks()
        self._clear_buffer()

    def _get_unit_importance(self):
        eps = 1e-10
        with torch.no_grad():
            for m in self.modules:
                w = fetch_mat_weights(m, False)  # output_dim * input_dim
                F_diag = (self.Fisher[m] / self.steps + eps)
                w_imp = w ** 2 * F_diag
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

    def kfac_step(self):
        with torch.no_grad():
            for m in self.modules:
                A, DS = self.A[m], self.DS[m]
                grad_mat = self.MatGradHandler(A, DS, m)
                if self.batch_averaged:
                    grad_mat *= DS.size(0)
                if self.steps == 0:
                    self.Fisher[m] = grad_mat.new(grad_mat.size()[1:]).fill_(0)
                self.Fisher[m] += (grad_mat.pow_(2)).sum(0)
                self.A[m] = None
                self.DS[m] = None
        self.steps += 1

    def _clear_buffer(self):
        self.Fisher = {}
        self.modules = []

    def estimate_importance(self, group, ch_groups=1) -> torch.Tensor:
        return self.importance(group, self.importances, ch_groups=ch_groups)

