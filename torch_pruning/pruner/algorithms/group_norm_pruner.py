import torch
import torch.nn as nn
import typing 
from .metapruner import MetaPruner
from .scheduler import linear_scheduler
from .. import function
from ..._helpers import _FlattenIndexMapping
from ... import ops

class GroupNormPruner(MetaPruner):
    """DepGraph: Towards Any Structural Pruning. 
    https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html
    """
    def __init__(
        self,
        model: nn.Module, # a simple pytorch model
        example_inputs: torch.Tensor, # a dummy input for graph tracing. Should be on the same 
        importance: typing.Callable, # tp.importance.Importance for group importance estimation
        reg=1e-4, # regularization coefficient
        alpha=4, # regularization scaling factor, [2^0, 2^alpha]
        global_pruning: bool = False, # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        pruning_ratio: float = 0.5,  # channel/dim pruning ratio, also known as pruning ratio
        pruning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific pruning ratio, will cover pruning_ratio if specified
        max_pruning_ratio: float = 1.0, # maximum pruning ratio. useful if over-pruning happens.
        iterative_steps: int = 1,  # for iterative pruning
        iterative_pruning_ratio_scheduler: typing.Callable = linear_scheduler, # scheduler for iterative pruning.
        ignored_layers: typing.List[nn.Module] = None, # ignored layers
        round_to: int = None,  # round channels to the nearest multiple of round_to

        # Advanced
        in_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer input
        out_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer output
        num_heads: typing.Dict[nn.Module, int] = dict(), # The number of heads for multi-head attention
        prune_num_heads: bool = False, # remove entire heads in multi-head attention
        prune_head_dims: bool = True, # remove head dimensions in multi-head attention
        head_pruning_ratio: float = 0.0, # head pruning ratio
        head_pruning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific head pruning ratio
        customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None, # pruners for customized layers. E.g., {nn.Linear: my_linear_pruner}
        unwrapped_parameters: typing.Dict[nn.Parameter, int] = None, # unwrapped nn.Parameters & pruning_dims. For example, {ViT.pos_emb: 0}
        root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
        forward_fn: typing.Callable = None, # a function to execute model.forward
        output_transform: typing.Callable = None, # a function to transform network outputs

        # deprecated
        channel_groups: typing.Dict[nn.Module, int] = dict(), # channel groups for layers
        ch_sparsity: float = None,
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None, 
    ):
        super(GroupNormPruner, self).__init__(
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
        self._groups = list(self.DG.get_all_groups(root_module_types=self.root_module_types, ignored_layers=self.ignored_layers))
        self.cnt = 0

    def step(self, interactive=False): 
        super(GroupNormPruner, self).step(interactive=interactive)
        # update the group list after pruning
        self._groups = list(self.DG.get_all_groups(root_module_types=self.root_module_types, ignored_layers=self.ignored_layers))

    @torch.no_grad()
    def regularize(self, model, alpha=2**4, bias=False):
        for i, group in enumerate(self._groups):
            ch_groups = self._get_channel_groups(group)
            imp = self.estimate_importance(group).sqrt()
            gamma = alpha**((imp.max() - imp) / (imp.max() - imp.min()))

            # Update Gradient
            for i, (dep, idxs) in enumerate(group):
                layer = dep.target.module
                prune_fn = dep.handler
                if prune_fn in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    if layer.weight.grad is None: continue

                    root_idxs = group[i].root_idxs
                    _gamma = torch.index_select(gamma, 0, torch.tensor(root_idxs, device=gamma.device))

                    w = layer.weight.data[idxs]
                    g = w * _gamma.view( -1, *([1]*(len(w.shape)-1)) ) #/ group_norm.view( -1, *([1]*(len(w.shape)-1)) ) * group_size #group_size #* gamma.view( -1, *([1]*(len(w.shape)-1)) )
                    layer.weight.grad.data[idxs]+=self.reg * g 
                    
                    if bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        g = b * _gamma
                        layer.bias.grad.data[idxs]+=self.reg * g 

                elif prune_fn in [
                    function.prune_conv_in_channels,
                    function.prune_linear_in_channels,
                ]:
                    if layer.weight.grad is None: continue
                    gn = imp
                    if hasattr(dep.target, 'index_transform') and isinstance(dep.target.index_transform, _FlattenIndexMapping):
                        gn = imp.repeat_interleave(w.shape[1]//imp.shape[0])
                    
                    # regularize input channels
                    if prune_fn==function.prune_conv_in_channels and layer.groups>1:
                        gamma = gamma[:len(idxs)//ch_groups]
                        idxs = idxs[:len(idxs)//ch_groups]

                    root_idxs = group[i].root_idxs
                    _gamma = torch.index_select(gamma, 0, torch.tensor(root_idxs, device=gamma.device))

                    w = layer.weight.data[:, idxs]
                    g = w * _gamma.view( 1, -1, *([1]*(len(w.shape)-2))  ) #/ gn.view( 1, -1, *([1]*(len(w.shape)-2)) ) * group_size #* gamma.view( 1, -1, *([1]*(len(w.shape)-2))  )
                    layer.weight.grad.data[:, idxs]+=self.reg * g
                    
                elif prune_fn == function.prune_batchnorm_out_channels:
                    # regularize BN
                    if layer.affine is not None:
                        if layer.weight.grad is None: continue

                        root_idxs = group[i].root_idxs
                        _gamma = torch.index_select(gamma, 0, torch.tensor(root_idxs, device=gamma.device))

                        w = layer.weight.data[idxs]
                        g = w * _gamma #/ group_norm * group_size
                        layer.weight.grad.data[idxs]+=self.reg * g 
                        
                        if bias and layer.bias is not None:
                            b = layer.bias.data[idxs]
                            g = b * _gamma #/ group_norm * group_size
                            layer.bias.grad.data[idxs]+=self.reg * g 
        self.cnt+=1