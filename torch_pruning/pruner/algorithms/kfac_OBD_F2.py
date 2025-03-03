"""
F2 = A ⊗ B,
A = in_c * in_c
B = out_c * out_c  (Diagonal)
"""
import torch

from .kfac_pruner import KFACMetaPruner
from ..kfac_utils.common_utils import tensor_to_list
from ..kfac_utils.kfac_utils import (ComputeMatGrad,
                              fetch_mat_weights,
                              input_to_raw_view)


class KFACOBDF2Pruner(KFACMetaPruner):
    def __init__(self, *args, **kwargs):
        super(KFACOBDF2Pruner, self).__init__(*args, **kwargs)
        print("Using OBD F2.")

    def _get_unit_importance(self):
        with torch.no_grad():
            for m in self.modules:
                w = fetch_mat_weights(m, False)  # output_dim * input_dim
                out_neuron_imp = (w**2 @ self.m_aa[m]).sum(1) * torch.diag(self.m_gg[m])
                in_neuron_imp = (self.m_gg[m] @ w**2).sum(0) * torch.diag(self.m_aa[m])
                in_neuron_imp = input_to_raw_view(in_neuron_imp, m)
                if len(in_neuron_imp.shape) == 3:
                    in_neuron_imp = in_neuron_imp.sum((1, 2))
                in_neuron_imp = in_neuron_imp.to('cpu')
                out_neuron_imp = out_neuron_imp.to('cpu')
                self.W_pruned[m] = w
                if m in self.importances.keys():
                    self.importances[m]['output'] = torch.cat([self.importances[m]['output'], out_neuron_imp.unsqueeze(0)],
                                                              dim=0)
                    self.importances[m]['input'] = torch.cat([self.importances[m]['input'], in_neuron_imp.unsqueeze(0)],
                                                                dim=0)
                else:
                    self.importances[m] = {'output': out_neuron_imp.unsqueeze(0)}
                    self.importances[m]['input'] = in_neuron_imp.unsqueeze(0)
                if len(self.importances[m]['input']) > self.record_length:
                    self.importances[m]['input'] = self.importances[m]['input'][1:]
                    self.importances[m]['output'] = self.importances[m]['output'][1:]

    def obtain_importance(self, dataloader, criterion, device, iter_steps=1000, fisher_type='true'):
        self._prepare_model()
        self.init_step()

        temp_loader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=64, shuffle=True,
                                                  num_workers=2)
        self._compute_fisher(temp_loader, criterion, device, fisher_type, iter_steps)

        self._get_unit_importance()
        self._rm_hooks()
        self._clear_buffer()