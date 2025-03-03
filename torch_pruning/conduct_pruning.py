import torch
import torch_pruning as tp
from tqdm import tqdm
import thop
from torch.nn.utils import prune
import time

def layerwise_computation(model, input_data):
    _ = model(input_data)
    flops_per_layer = []
    params_per_layer = []
    input_neurons_per_layer = []
    output_neurons_per_layer = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            macs, params = thop.profile(module, inputs=(module.X,))
            params = module.weight.numel() + module.bias.numel() if module.bias is not None else module.weight.numel()
            flops_per_layer.append(macs)
            params_per_layer.append(params)
            input_neurons_per_layer.append(module.X.shape[1])
            output_neurons_per_layer.append(module.Y.shape[1])
    return torch.tensor(flops_per_layer), torch.tensor(params_per_layer), torch.tensor(input_neurons_per_layer), torch.tensor(output_neurons_per_layer)

class WrappedPruner:
    def __init__(self, args, train_loader, test_loader, model, example_inputs, ignored_layers, pruning_ratio, device, speed_up=1000, builder=None, base_ops=None, base_params=None):
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.importance_type = args.importance_type
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.example_inputs = example_inputs
        self.iters_per_step = args.iters_per_step
        self.pruning_ratio = pruning_ratio
        self.iterative_steps = args.iterative_steps
        self.device = device
        self.speed_up = speed_up
        model.eval()
        if args.importance_type == "OBA":
            imp = tp.oba_importance.HessianImportance(normalizer=args.normalizer, multivariable=args.multivariable)
            # imp = tp.oba_importance.HessianWeightImportance(normalizer=args.normalizer, multivariable=args.multivariable)
            pruner = tp.pruner.OBAPruner(
                model,
                example_inputs,
                global_pruning=True,
                importance=imp,
                iterative_steps=args.iterative_steps,
                pruning_ratio=pruning_ratio,
                pruning_ratio_dict={},
                max_pruning_ratio=args.max_pruning_ratio,
                ignored_layers=ignored_layers,
                unwrapped_parameters=[],
                delta=args.delta,
                upward_delta=args.upward_delta,
                downward_delta=args.downward_delta,
                parallel_delta=args.parallel_delta,
                self_unit_weight=args.self_unit_weight,
                other_unit_weight=args.other_unit_weight
            )
            pruner.register_hooks()
        elif "OBD" in args.importance_type or "OBS" in args.importance_type or "Eigen" in args.importance_type:
            imp = tp.kfac_importance.PostLayerKFACImportance(normalizer=args.normalizer, multivariable=args.multivariable)
            if args.importance_type == "C-OBD":
                pruner_handler = tp.pruner.FisherDiagPruner
            elif args.importance_type == "C-OBS":
                pruner_handler = tp.pruner.KFACMetaPruner
            elif args.importance_type == "Eigen":
                pruner_handler = tp.pruner.KFACEigenPruner
            elif args.importance_type == "Kron-OBD":
                pruner_handler = tp.pruner.KFACOBDF2Pruner
            elif args.importance_type == "Kron-OBS":
                pruner_handler = tp.pruner.KFACOBSF2Pruner
            else:
                raise ValueError("Importance type not recognized.")
            if args.importance_type == "Eigen":
                pruner = pruner_handler(
                    model,
                    builder,
                    example_inputs,
                    global_pruning=True,
                    importance=imp,
                    iterative_steps=args.iterative_steps,
                    pruning_ratio=pruning_ratio,
                    pruning_ratio_dict={},
                    ignored_layers=ignored_layers,
                    unwrapped_parameters=[],
                )
            else:
                pruner = pruner_handler(
                    model,
                    example_inputs,
                    global_pruning=True,
                    importance=imp,
                    iterative_steps=args.iterative_steps,
                    pruning_ratio=pruning_ratio,
                    pruning_ratio_dict={},
                    ignored_layers=ignored_layers,
                    unwrapped_parameters=[],
                )

        else:
            if args.importance_type == "Weight":
                imp = tp.importance.MagnitudeImportance(p=1, normalizer=args.normalizer)
            elif args.importance_type == "Taylor":
                imp = tp.importance.TaylorImportance(normalizer=args.normalizer, multivariable=args.multivariable)
            else:
                raise ValueError("Importance type not recognized.")
            pruner = tp.pruner.MagnitudePruner(
                model,
                example_inputs,
                global_pruning=True,
                importance=imp,
                iterative_steps=args.iterative_steps,
                pruning_ratio=pruning_ratio,
                ignored_layers=ignored_layers,
            )
        with torch.no_grad():
            if base_ops is None:
                self.base_ops, self.base_params = thop.profile(model, inputs=(example_inputs,))
            else:
                self.base_ops = base_ops
                self.base_params = base_params
        self.pruner = pruner
        self.finish_prune = False

    def iterative_prune_step(self):
        ## prune
        # self.pruner.register_hooks()
        device = self.device
        self.model.train()
        cross_entropy = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        optimizer.zero_grad()
        prune_iter = 0
        t1 = time.time()
        if "OBD" in self.importance_type or "OBS" in self.importance_type or "Eigen" in self.importance_type:
            self.pruner.obtain_importance(self.train_loader, cross_entropy, device, iter_steps=self.iters_per_step)
        else:
            for img, label in tqdm(self.train_loader, total=self.iters_per_step - 1):
                prune_iter += 1
                img = img.to(device)
                label = label.to(device)
                loss = cross_entropy(self.model(img), label)
                if self.importance_type == "OBA":
                    optimizer.zero_grad()
                    self.pruner.obtain_importance(loss)
                    loss.backward()
                    for module in self.model.modules():
                        if hasattr(module, "X"):
                            del module.X
                            del module.Y
                            del module.grad_input
                            del module.grad_output
                else:
                    loss.backward()
                if prune_iter >= self.iters_per_step:
                    break
        t2 = time.time()
        self.pruner.step(interactive=False)
        t3 = time.time()
        for module in self.model.modules():
            if hasattr(module, "X"):
                del module.X
                del module.Y
            if hasattr(module, "grad_input"):
                del module.grad_input
                del module.grad_output
        # pruned_ops, pruned_params = tp.utils.count_ops_and_params(self.model, example_inputs=self.example_inputs)
        with torch.no_grad():
            pruned_ops, pruned_params = thop.profile(self.model, inputs=(self.example_inputs,))
        ops_percent = pruned_ops / self.base_ops
        params_percent = pruned_params / self.base_params
        current_speed_up = 1 / ops_percent
        if self.importance_type == "OBA":
            self.pruner.update_dependency_graph(self.example_inputs)
            self.pruner.group_importances = self.pruner.initialize_importance()
        if "OBD" in self.importance_type or "OBS" in self.importance_type or "Eigen" in self.importance_type:
            self.pruner.importances = {}
        if self.pruner.current_step == self.pruner.iterative_steps or current_speed_up >= self.speed_up:
            self.finish_prune = True
        for module in self.model.modules():
            if hasattr(module, "X"):
                del module.X
                del module.Y
        return ops_percent, params_percent#, t2 - t1, t3 - t2

    def onepass_prune_step(self, ops_ratio):
        ## prune
        device = self.device
        self.model.train()
        cross_entropy = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        optimizer.zero_grad()
        prune_iter = 0
        if "OBD" in self.importance_type or "OBS" in self.importance_type or "Eigen" in self.importance_type:
            self.pruner.obtain_importance(self.train_loader, cross_entropy, device, iter_steps=self.iterative_steps)
        else:
            for img, label in tqdm(self.train_loader, total=min(self.iters_per_step - 1, len(self.train_loader))):
                prune_iter += 1
                img = img.to(device)
                label = label.to(device)
                loss = cross_entropy(self.model(img), label)
                if self.importance_type == "OBA":
                    optimizer.zero_grad()
                    self.pruner.obtain_importance(loss)
                    loss.backward()
                    for module in self.model.modules():
                        if hasattr(module, "X"):
                            del module.X
                            del module.Y
                            del module.grad_input
                            del module.grad_output
                else:
                    loss.backward()
                if prune_iter >= self.iters_per_step:
                    break
        ops_percent = 1
        while ops_ratio < ops_percent:
            print(ops_percent)
            self.pruner.step(interactive=False)
            for module in self.model.modules():
                if hasattr(module, "X"):
                    del module.X
                    del module.Y
                if hasattr(module, "grad_input"):
                    del module.grad_input
                    del module.grad_output
            with torch.no_grad():
                pruned_ops, pruned_params = thop.profile(self.model, inputs=(self.example_inputs,))
            ops_percent = pruned_ops / self.base_ops
            params_percent = pruned_params / self.base_params
            current_speed_up = 1 / ops_percent
            if self.pruner.current_step > self.pruner.iterative_steps:
                break
        if self.importance_type == "OBA":
            self.pruner.update_dependency_graph(self.example_inputs)
            self.pruner.group_importances = self.pruner.initialize_importance()
        if "OBD" in self.importance_type or "OBS" in self.importance_type or "Eigen" in self.importance_type:
            self.pruner.importances = {}
            if self.importance_type == "Eigen":
                self.pruner._clear_buffer()
        if self.pruner.current_step == self.pruner.iterative_steps or current_speed_up >= self.speed_up:
            self.finish_prune = True
        for module in self.model.modules():
            if hasattr(module, "X"):
                del module.X
                del module.Y
        return ops_percent, params_percent

    def onepass_unstructured_prune_step(self, pruning_ratio):
        for module in self.model.modules():
            if hasattr(module, "weight_mask"):
                module.weight_orig.data = module.weight_mask * module.weight_orig.data
                prune.remove(module, "weight")
            if hasattr(module, "bias_mask"):
                module.bias_orig.data = module.bias_mask * module.bias_orig.data
                prune.remove(module, "bias")
        ## prune
        device = self.device
        self.model.train()
        cross_entropy = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        optimizer.zero_grad()
        prune_iter = 0
        for img, label in tqdm(self.train_loader, total=min(self.iters_per_step - 1, len(self.train_loader))):
            prune_iter += 1
            img = img.to(device)
            label = label.to(device)
            loss = cross_entropy(self.model(img), label)
            if self.importance_type == "OBA":
                optimizer.zero_grad()
                self.pruner.obtain_importance(loss)
                loss.backward()
                for module in self.model.modules():
                    if hasattr(module, "X"):
                        del module.X
                        del module.Y
                        del module.grad_input
                        del module.grad_output
            else:
                loss.backward()
            if prune_iter > self.iters_per_step:
                break
        mask_dict = self.pruner.unstructured_prune(pruning_ratio)
        # mask_dict = self.pruner.layerwise_unstructured_prune(pruning_ratio)
        for module in self.model.modules():
            if hasattr(module, "X"):
                del module.X
                del module.Y
            if hasattr(module, "grad_input"):
                del module.grad_input
                del module.grad_output
        return mask_dict


    def remove_hooks(self):
        if hasattr(self.pruner, "remove_hooks"):
            self.pruner.remove_hooks()