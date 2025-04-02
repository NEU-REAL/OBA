import os
import torch
import torch.nn as nn
import numpy as np
import math

def forward_hook(self, input, output):
    if torch.tensor(list(map(lambda x:isinstance(x, torch.Tensor), input))).sum() > 1:
        self.X = []
        for x in input:
            if isinstance(x, torch.Tensor):
                self.X.append(x)
    else:
        self.X = input[0]
    self.Y = output

def cosine_sim(data1, data2):
    cosine_similarity = torch.nn.CosineSimilarity(dim=0)
    data1 = (data1 - data1.min()) / (data1.max() - data1.min())
    data2 = (data2 - data2.min()) / (data2.max() - data2.min())
    return cosine_similarity(data1, data2)


def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class PruneLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PruneLayer, self).__init__(*args, **kwargs)
        self.up_hessian = None
        self.down_hessian = None
        self.up_taylor = None
        self.down_taylor = None
        self.count = 0
        self.register_forward_hook(forward_hook)
        self.register_full_backward_hook(backward_hook)

    def surrogate_forward(self, *args, **kwargs):
        pass


class Linear(PruneLayer, nn.Linear):
    def get_taylor(self, parameters, num_up_layers):
        list_parameters = [param for param in parameters]
        output_grad = self.grad_output[0]
        B = output_grad.shape[0]
        batched_output_grad = torch.zeros_like(output_grad)
        batched_output_grad = batched_output_grad.unsqueeze(0).repeat(batched_output_grad.shape[1], *([1] * len(batched_output_grad.shape)))
        for i in range(batched_output_grad.shape[0]):
            batched_output_grad[i, :, i] = output_grad[:, i]
        input_grad = self.grad_input[0]
        batched_input_grad = torch.zeros_like(input_grad)
        batched_input_grad = batched_input_grad.unsqueeze(0).repeat(batched_input_grad.shape[1], *([1] * len(batched_input_grad.shape)))
        for i in range(batched_input_grad.shape[0]):
            batched_input_grad[i, :, i] = input_grad[:, i]



        output_gradients = torch.autograd.grad(self.Y, list_parameters, batched_output_grad, retain_graph=True, allow_unused=True, is_grads_batched=True)
        up_downward_hessian = torch.stack(
            [(tensor * list_parameters[idx].data).view(tensor.shape[0], -1).sum(1) if tensor is not None else torch.zeros(batched_output_grad.shape[0]).to(batched_output_grad.device) for
             idx, tensor in enumerate(output_gradients)], dim=0).sum(0)
        input_gradients = torch.autograd.grad(self.X, list_parameters, batched_input_grad, retain_graph=True, allow_unused=True, is_grads_batched=True)
        down_downward_hessian = torch.stack(
            [(tensor * list_parameters[idx].data).view(tensor.shape[0], -1).sum(1) if tensor is not None else torch.zeros(batched_input_grad.shape[0]).to(batched_input_grad.device) for
             idx, tensor in enumerate(input_gradients)], dim=0).sum(0)

        weight_taylor = torch.autograd.grad(self.Y, self.weight, output_grad, retain_graph=True, allow_unused=True)[0] * self.weight
        bias_taylor = torch.autograd.grad(self.Y, self.bias, output_grad, retain_graph=True, allow_unused=True)[0] * self.bias
        up_upward_hessian = (weight_taylor.sum(dim=1)) * (num_up_layers + 1)
        down_upward_hessian = weight_taylor.sum(dim=0) * (num_up_layers + 1)

        up_taylor = up_upward_hessian.abs()
        down_taylor = down_upward_hessian.abs()
        up_hessian = (up_upward_hessian + up_downward_hessian).abs()
        down_hessian = (down_upward_hessian + down_downward_hessian).abs()
        if num_up_layers == 0:
            print(cosine_sim(up_upward_hessian, up_downward_hessian), "upper layer {}".format(num_up_layers))
        if self.up_hessian is not None:
            self.up_taylor += up_taylor.detach().to(self.up_taylor.device).to(torch.float64)
            self.down_taylor += down_taylor.detach().to(self.down_taylor.device).to(torch.float64)
            self.up_hessian += up_hessian.detach().to(self.up_hessian.device).to(torch.float64)
            self.down_hessian += down_hessian.detach().to(self.down_hessian.device).to(torch.float64)
        else:
            self.up_hessian = up_hessian.detach().to(torch.float64).to("cpu")
            self.down_hessian = down_hessian.detach().to(torch.float64).to("cpu")
            self.up_taylor = up_taylor.detach().to(torch.float64).to("cpu")
            self.down_taylor = down_taylor.detach().to(torch.float64).to("cpu")
        self.count += 1

    def surrogate_forward(self, x, weight):
        return torch.nn.functional.linear(x, weight, self.bias)



class WrappedMLP(nn.Module):
    def __init__(self, shp):
        super().__init__()
        self.in_dim = shp[0]
        self.out_dim = shp[-1]
        self.depth = len(shp) - 1
        linear_list = []
        for i in range(self.depth):
            linear_list.append(Linear(shp[i], shp[i + 1]))
            linear_list.append(nn.SiLU())
        self.linears = nn.ModuleList(linear_list)

    def forward(self, x, mask_layer=None, layer_idx=None):
        x = x.view(x.shape[0], -1)
        if mask_layer is not None:
            for layer in self.linears[:mask_layer]:
                x = layer(x)
            x[:, layer_idx] = 0
            for layer in self.linears[mask_layer:]:
                x = layer(x)
        else:
            for layer in self.linears:
                x = layer(x)
        return x

    def get_taylor(self):
        for idx, layer in enumerate(self.linears):
            if hasattr(layer, 'X'):
                layer.get_taylor(self.parameters(), self.depth - idx // 2 - 1)

    def reset_taylor(self):
        for layer in self.modules():
            if hasattr(layer, 'up_hessian'):
                layer.up_hessian = None
                layer.down_hessian = None
                layer.count = 0

class NNMLP(nn.Module):
    def __init__(self, shp):
        super().__init__()
        self.in_dim = shp[0]
        self.out_dim = shp[-1]
        self.depth = len(shp) - 1
        linear_list = []
        for i in range(self.depth):
            linear_list.append(nn.Linear(shp[i], shp[i + 1]))
            linear_list.append(nn.SiLU())
        self.linears = nn.ModuleList(linear_list)

    def forward(self, x, mask_layer=None, layer_idx=None):
        # x = x.view(x.shape[0], -1)
        if mask_layer is not None:
            for layer in self.linears[:mask_layer]:
                x = layer(x)
            x[:, layer_idx] = 0
            for layer in self.linears[mask_layer:]:
                x = layer(x)
        else:
            for layer in self.linears:
                x = layer(x)
        return x

    def get_taylor(self):
        for idx, layer in enumerate(self.linears):
            if hasattr(layer, 'X'):
                layer.get_taylor(self.parameters(), self.depth - idx // 2 - 1)

    def reset_taylor(self):
        for layer in self.modules():
            if hasattr(layer, 'up_hessian'):
                layer.up_hessian = None
                layer.down_hessian = None
                layer.count = 0


class ConvNet(nn.Module):
    def __init__(self, channel_shape, resolution, class_num):
        super().__init__()
        self.in_dim = channel_shape[0]
        self.out_dim = channel_shape[-1]
        self.depth = len(channel_shape) - 1
        conv_list = []
        self.resolution = resolution
        for i in range(self.depth):
            conv_list.append(nn.Conv2d(channel_shape[i], channel_shape[i + 1], kernel_size=3, stride=1, padding=1, bias=True))
            conv_list.append(nn.SiLU())
        self.convs = nn.ModuleList(conv_list)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.linear = nn.Linear(np.prod((*resolution, channel_shape[-1])), class_num)


    def forward(self, x, mask_layer=None, layer_idx=None):
        if mask_layer is not None:
            for layer in self.convs[:mask_layer]:
                x = layer(x)
            x[:, layer_idx] = 0
            for layer in self.convs[mask_layer:]:
                x = layer(x)
        else:
            for layer in self.convs:
                x = layer(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Attention(nn.Module):
    def __init__(self, num_heads=2, embed_dim=64, record_attn=False):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.record_attn = record_attn
        self.attn = None

    def forward(self, q, k, v):
        b, n, _ = q.shape
        # reshape q, k, v for multihead attention and make em batch first
        q = q.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(2, 3)) / torch.sqrt(torch.tensor(self.head_dim))
        attn = torch.softmax(attn, dim=-1)
        if self.record_attn:
            self.attn = attn
        out = attn @ v
        out = out.reshape(b, self.num_heads, n, self.head_dim).permute(0, 2, 1, 3).reshape(b, n, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        return out


class Block(nn.Module):
    # A transformer block
    def __init__(self, n_head=2, n_embed=6, self_attn=False):
        super().__init__()
        self.n_head = n_head
        self.n_embed = n_embed
        self.self_attn = self_attn
        self.ln_1 = nn.LayerNorm(n_embed)
        if not self.self_attn:
            self.qkv_mlp = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.attn = Attention(num_heads=n_head, embed_dim=n_embed, record_attn=True)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = NNMLP(shp=[n_embed, 4 * n_embed, n_embed])

    def forward(self, x):
        b, n, c = x.shape
        if self.self_attn:
            q, k, v = x, x, x
        else:
            qkv = self.qkv_mlp(x).reshape(
                b, n, 3, self.n_embed).permute(2, 0, 1, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]
        x = x + self.attn(q, k, v)
        x = x + self.mlp(x)
        return x

    def get_linear_layers(self):
        return [self.ln_1, *self.attn.get_linear_layers(), self.ln_2, *self.mlp.get_linear_layers()]


class Transformer(nn.Module):
    # Transformer: since our goal is to deal with linear regression, not language,
    # we ignore token embeddings and positioanl embeddings.
    def __init__(self, in_dim=3, out_dim=3, n_head=2, n_embed=20, n_layer=2, block_size=19, self_attn=False):
        super().__init__()
        self.n_head = n_head
        self.n_embed = n_embed
        self.n_layer = n_layer
        self.l_i = nn.Linear(in_dim, n_embed)
        self.blocks = nn.ModuleList([Block(n_head=n_head, n_embed=n_embed, self_attn=self_attn) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        #self.ln_f.weight.requires_grad = False
        #self.ln_f.bias.requires_grad = False
        self.l_f = nn.Linear(n_embed, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x, mask_idx=None):
        x = self.l_i(x)
        if mask_idx is not None:
            x[:, :, mask_idx] = 0
        # x = x + self.pe.unsqueeze(dim=0) # positional encoding
        for i in range(self.n_layer):
            x = self.blocks[i](x)
        # x = self.ln_f(x)
        y = self.l_f(x)
        return y