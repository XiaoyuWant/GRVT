# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np


import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.transforms import Resize


from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs

from .modeling_resnet import ResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs ##if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        # if config.patches.get("grid") is not None:
        #     grid_size = config.patches["grid"]
        #     patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        #     n_patches = (img_size[0] // 16) * (img_size[1] // 16)
        #     self.hybrid = True
        # else:
        #     patch_size = _pair(config.patches["size"])
        #     n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        #     self.hybrid = False

        # if self.hybrid:
        #     self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
        #                                  width_factor=config.resnet.width_factor)
        #     in_channels = self.hybrid_model.width * 16
        # self.patch_embeddings = Conv2d(in_channels=in_channels,
        #                                out_channels=config.hidden_size,
        #                                kernel_size=patch_size,
        #                                stride=patch_size)
        # patch_size = _pair(config.patches["size"])
        # n_patches = ((img_size[0] - patch_size[0]) // 8 + 1) * ((img_size[1] - patch_size[1]) // 8 + 1)
        # self.patch_embeddings = Conv2d(in_channels=in_channels,
        #                                 out_channels=config.hidden_size,
        #                                 kernel_size=patch_size,
        #                                 stride=(8, 8))
        self.scale = config.scale
        patch_size = _pair(config.patches["size"])
        if config.split == 'non-overlap':
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            # print(n_patches)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
            if self.scale==True:
                scale_patches=(img_size[0]//2 // patch_size[0]) * (img_size[1]//2 // patch_size[1])
                # print(n_patches)
                self.scale_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
                self.scale_position_embeddings = nn.Parameter(torch.zeros(1, scale_patches+1, config.hidden_size))
                self.cls_token_scale = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

            # self.g_pool=torch.nn.AdaptiveAvgPool2d((14,14))
        elif config.split == 'overlap':
            n_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * ((img_size[1] - patch_size[1]) // config.slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                        out_channels=config.hidden_size,
                                        kernel_size=patch_size,
                                        stride=(config.slide_step, config.slide_step))

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        y=x
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # print(x.shape)
        if self.hybrid:
            x = self.hybrid_model(x)
        # print(x.shape)
        # g_x=self.global_embeddings(x)
        # g_x=self.g_pool(g_x)
        # print(g_x.shape)
        x = self.patch_embeddings(x)  #x=[16,768,14,14]
        x = x.flatten(2)
        x = x.transpose(-1, -2) #->[16,196,768]
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        
        #resize
        if self.scale==True:
            torch_resize = Resize([112,112])
            y = torch_resize(y)
            y = self.scale_embeddings(y)
            y = y.flatten(2)
            #x=torch.cat([x,y],2)
            y = y.transpose(-1, -2) #->[16,196,768]
            cls_token_scale=self.cls_token_scale.expand(B, -1, -1)
            y = torch.cat((cls_token_scale, y), dim=1)
            scale_embeddings = y + self.scale_position_embeddings

            # print(scale_embeddings.shape)
            scale_embeddings=torch.cat((embeddings,scale_embeddings),dim=1)
            scale_embeddings = self.dropout(scale_embeddings)
            
            return scale_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)

        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        #print(x.shape) x [16,197,768]
        #print(weights.shape) [16,12,197,197]
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class AttentionMix(nn.Module):
    def __init__(self):
        super(AttentionMix, self).__init__()

    def forward(self, x,topn):
        length = len(x)# x=[11layer,batchsize,12=attention_heads,197,197] 197=hideen_size+1
        # print(length)
        last_map = x[0] #最后一个weights，
        # print(last_map.shape)
        batchsize=last_map.shape[0]
        # print(last_map.shape)
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        # last_map = last_map[:,:,0,1:] # 应该是跟last_map[:,:,1:,0]等效的->[16,12,197]
        last_map0= last_map[:,:,0,1:197]
        last_map1= last_map[:,:,0,198:]
        
        #选择阈值？最终得到 [16,num_you_want]个结果
        # torch.sort(last_map,dim=2,descending=True)
        # # print(last_map.shape)
        # last_map=last_map[:,:,:10]#取10
        # last_map=last_map.reshape(16,120)
        maxk=max((topn,))
        _,pred0=last_map0.topk(maxk,dim=2)
        _,pred1=last_map1.topk(maxk,dim=2)
        
        last_map0=pred0.reshape(batchsize,12*topn)
        last_map1=pred1.reshape(batchsize,12*topn)
        # print(last_map0.shape)
        # print(last_map1.shape)
        last_map0=last_map0+1
        last_map1=last_map1+198 #[16,25]
        last_map=torch.cat([last_map0,last_map1],dim=1)
        # print(last_map.shape)



        # _, max_inx = last_map.max(2) #返回 tensor_max tensor_index，意思是每个head选择了196个部分中的一个

        return last_map


class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def forward(self, x):
        length = len(x)# x=[12layer,batchsize,num_head,197,197] 197=hideen_size+1
        last_map = x[0] #最后一个weights，
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        last_map = last_map[:,:,0,1:] # 应该是跟last_map[:,:,1:,0]等效的->[batchsize,numhead,196]

        _, max_inx = last_map.max(2) #返回 tensor_max tensor_index
        # print(max_inx)
        # print(max_inx.shape)#[16,12]
        return _, max_inx

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.mixrank=config.mixrank
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(11):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))
        

        self.mix_attention=AttentionMix()
        self.part_attention=Part_Attention()

        self.layer2 = nn.ModuleList()
        # self.encoder_norm2 = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(1):
            layer = Block(config, vis)
            self.layer2.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attn_weights.append(weights)
        topn=3 # 1 3 5
        for layer_block in self.layer2:
            if self.mixrank:
                parts=[]
                last_map=self.mix_attention(attn_weights,topn)
                B,num=last_map.shape
                for i in range(B):
                    parts.append(hidden_states[i,last_map[i,:]])
                parts = torch.stack(parts).squeeze(1)
                # hidden_states = torch.cat((hidden_states[:,0].unsqueeze(1),parts), dim=1)
                hidden_states = torch.cat((hidden_states[:,0].unsqueeze(1),hidden_states[:,197].unsqueeze(1),parts), dim=1)

            hidden_states, weights = layer_block(hidden_states)
            # # print("hidden_states.shape:",hidden_states.shape)
            # attn_weights.append(weights)
        # print(hidden_states.shape) #[16, 197, 768] [16, 50, 768]

        encoded = self.encoder_norm(hidden_states)
        # print(encoded.shape)
        return encoded, attn_weights
        #return hidden_states


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        # encoded_scale,attn_weights_scale = self.encoder(scale_embedding_output)

        
        return encoded, attn_weights#,encoded_scale,attn_weights_scale


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.config=config
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        
        config.split='non-overlap'
        

    

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)
        # self.head_scale = Linear(config.hidden_size, num_classes)
        # self.head2 = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x) #y,attn_weights_scale
        #print(attn_weights)
        logits = self.head(x[:, 0])
        if self.config.scale==True:
            if self.config.mixrank:
                scale_token = 1
            else:
                scale_token = 197
            logits_scale = self.head(x[:,scale_token]) # 不用mix模块是197，用了是1
            logits=torch.mean(torch.stack([logits,logits_scale], dim=0), dim=0)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))  
            return loss
        else:
            return logits, attn_weights


    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            if self.config.scale:
                self.transformer.embeddings.scale_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
                self.transformer.embeddings.scale_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
                
            if self.config.scale:
                posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
                posemb_scale = self.transformer.embeddings.scale_position_embeddings
                if posemb.size() == posemb_scale.size():
                    self.transformer.embeddings.scale_position_embeddings.copy_(posemb)
                else:
                    logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_scale.size()))
                    ntok_new = posemb_scale.size(1)

                    if self.classifier == "token":
                        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                        ntok_new -= 1
                    else:
                        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                    gs_old = int(np.sqrt(len(posemb_grid)))
                    gs_new = int(np.sqrt(ntok_new))
                    print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                    posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                    posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                    self.transformer.embeddings.scale_position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
