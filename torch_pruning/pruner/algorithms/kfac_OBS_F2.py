"""
F2 = A ⊗ B,
A = in_c * in_c
B = out_c * out_c  (Diagonal)
"""
import torch

from .kfac_pruner import KFACMetaPruner
from ..kfac_utils.common_utils import tensor_to_list
from ..kfac_utils.kfac_utils import (fetch_mat_weights,
                              mat_to_weight_and_bias)


class KFACOBSF2Pruner(KFACMetaPruner):
    def __init__(self, *args, **kwargs):
        super(KFACOBSF2Pruner, self).__init__(*args, **kwargs)
        print("Using OBS F2.")

    def _get_unit_importance(self):
        eps = 1e-10
        with torch.no_grad():
            for m in self.modules:
                w = fetch_mat_weights(m, False)  # output_dim * input_dim
                A_inv = self.Q_a[m] @ (torch.diag(1.0 / (self.d_a[m] + eps))) @ self.Q_a[m].t()
                G_inv = self.Q_g[m] @ (torch.diag(1.0 / (self.d_g[m] + eps))) @ self.Q_g[m].t()
                out_neuron_imp = torch.sum(w**2@self.m_aa[m], 1) / torch.diag(G_inv)
                in_neuron_imp = torch.sum(self.m_gg[m]@w**2, 0) / torch.diag(A_inv)
                out_neuron_imp = out_neuron_imp.to('cpu')
                in_neuron_imp = in_neuron_imp.to('cpu')
                self.W_pruned[m] = w
                if m in self.importances.keys():
                    self.importances[m]['output'] = torch.cat(
                        [self.importances[m]['output'], out_neuron_imp.unsqueeze(0)],
                        dim=0)
                    self.importances[m]['input'] = torch.cat([self.importances[m]['input'], in_neuron_imp.unsqueeze(0)],
                                                             dim=0)
                else:
                    self.importances[m] = {'output': out_neuron_imp.unsqueeze(0)}
                    self.importances[m]['input'] = in_neuron_imp.unsqueeze(0)
                if len(self.importances[m]['input']) > self.record_length:
                    self.importances[m]['input'] = self.importances[m]['input'][1:]
                    self.importances[m]['output'] = self.importances[m]['output'][1:]


