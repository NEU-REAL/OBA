import numpy as np
import time
import torch
import torch.nn as nn
from sktensor import dtensor, cp_als
import torch.nn as nn
import typing
from ... import ops
from .. import function
from .kfac_eigen_pruner import KFACEigenPruner


def get_UDV_decomposition(W, method='svd'):
    # current implementation is svd
    c_out, khkw, c_in = W.shape
    method = method.lower()
    with torch.no_grad():
        if method == 'svd':
            m_W = W.mean(1)
            U, _, V = torch.svd(m_W)
            D = []
            for r in range(khkw):
                W_r = W[:, r, :]  # c_out * c_in  ->
                c = min(c_out, c_in)
                D_w = torch.diag(U.t() @ W_r @ V).view(c, 1)
                D.append(D_w)
            S = torch.stack(D, dim=1)
        elif method == 'svd_avg':
            pass
        elif method == 'als':
            # m = min(c_out, c_in)
            # U: c_out * m
            # S: k^2 * m
            # V: c_in * m
            rank = min(c_out, c_in)

            tic = time.clock()
            T = dtensor(W.data.cpu().numpy())
            P, fit, itr, exectimes = cp_als(T, rank, init='random')
            U = np.array(P.U[0])  # c_out * rank
            S = np.array(P.U[1]).T  # k^2 * rank --> rank * k^2
            V = np.array(P.U[2] * P.lmbda)  # c_in * rank
            print('CP decomposition done. It cost %.5f secs. fit: %f' % (time.clock() - tic, fit[0]))
            V = torch.FloatTensor(V).cuda()
            S = torch.FloatTensor(S).cuda()
            U = torch.FloatTensor(U).cuda()

        else:
            raise NotImplementedError("Method {} not supported!".format(method))

    return U, S, V


class KFACEigenSVDPruner(KFACEigenPruner):

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
                 max_pruning_ratio: float = 1.0,  # maximum pruning ratio. useful if over-pruning happens.
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

    def obtain_importance(self, dataloader, criterion, device, fisher_type='true'):
        self._prepare_model()
        self.init_step()

        self._compute_fisher(dataloader, criterion, device, fisher_type)
        self._update_inv()  # eigen decomposition of fisher

        self._get_unit_importance()
        self._merge_Qs()  # update the eigen basis
        self._make_depth_separable()

        self._rm_hooks()
        self._clear_buffer()


    def _make_depth_separable(self):
        assert self._inversed
        for idx, m in enumerate(self.remain_modules):
            if isinstance(m, nn.Conv2d):
                W_star = self.W_star[m]  # c_out * (kh * kw) * c_in
                m.groups = min(W_star.shape[-1], W_star.shape[0])
                U, W_sep, V = get_UDV_decomposition(W_star, method='als')
                try:
                    self.Q_a[m] = self.Q_a[m] @ V
                    self.Q_g[m] = self.Q_g[m] @ U
                    self.W_star[m] = W_sep
                except:
                    import pdb; pdb.set_trace()




        # self.remain_modules = []
        # for m in self.modules:
        #     self.remain_modules.append(m)


