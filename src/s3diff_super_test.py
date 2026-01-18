import os
import re
import requests
import sys
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_lora_fwd
from basicsr.archs.arch_util import default_init_weights
import torch.nn.functional as F

def get_layer_number(module_name):
    base_layers = {
        'down_blocks': 0,
        'mid_block': 4,
        'up_blocks': 5
    }

    if module_name == 'conv_out':
        return 9

    base_layer = None
    for key in base_layers:
        if key in module_name:
            base_layer = base_layers[key]
            break

    if base_layer is None:
        return None

    additional_layers = int(re.findall(r'\.(\d+)', module_name)[0]) #sum(int(num) for num in re.findall(r'\d+', module_name))
    final_layer = base_layer + additional_layers
    return final_layer


class OPT_S3Diff(torch.nn.Module):
    def __init__(self, sd_path=None, pretrained_path=None, lora_rank_unet=32, lora_rank_vae=16, block_embedding_dim=64):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(sd_path)

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")

        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]

        num_embeddings = 64
        # self.conv_W = nn.Sequential(
        #     nn.Conv2d(1, 3, kernel_size=3, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(3, 1, kernel_size=3, padding=1),
        #     nn.ReLU(True),
        # )
        self.W = nn.Linear(49, 256)

        self.vae_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )

        self.unet_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )

        self.vae_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.unet_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.vae_fuse_mlp = nn.Linear(256 + 64, lora_rank_vae ** 2)
        self.unet_fuse_mlp = nn.Linear(256 + 64, lora_rank_unet ** 2)

        default_init_weights([self.vae_de_mlp, self.unet_de_mlp, self.vae_block_mlp, self.unet_block_mlp, \
            self.vae_fuse_mlp, self.unet_fuse_mlp], 1e-5)

        # vae
        self.vae_block_embeddings = nn.Embedding(6, block_embedding_dim)
        self.unet_block_embeddings = nn.Embedding(10, block_embedding_dim)

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()

            _W = self.W.state_dict()
            for k in sd["state_dict_unet_fuse_W"]:
                _W[k] = sd["state_dict_unet_fuse_W"][k]
            self.W.load_state_dict(_W)

            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)

            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

            _vae_de_mlp = self.vae_de_mlp.state_dict()
            for k in sd["state_dict_vae_de_mlp"]:
                _vae_de_mlp[k] = sd["state_dict_vae_de_mlp"][k]
            self.vae_de_mlp.load_state_dict(_vae_de_mlp)

            _unet_de_mlp = self.unet_de_mlp.state_dict()
            for k in sd["state_dict_unet_de_mlp"]:
                _unet_de_mlp[k] = sd["state_dict_unet_de_mlp"][k]
            self.unet_de_mlp.load_state_dict(_unet_de_mlp)

            _vae_block_mlp = self.vae_block_mlp.state_dict()
            for k in sd["state_dict_vae_block_mlp"]:
                _vae_block_mlp[k] = sd["state_dict_vae_block_mlp"][k]
            self.vae_block_mlp.load_state_dict(_vae_block_mlp)

            _unet_block_mlp = self.unet_block_mlp.state_dict()
            for k in sd["state_dict_unet_block_mlp"]:
                _unet_block_mlp[k] = sd["state_dict_unet_block_mlp"][k]
            self.unet_block_mlp.load_state_dict(_unet_block_mlp)

            _vae_fuse_mlp = self.vae_fuse_mlp.state_dict()
            for k in sd["state_dict_vae_fuse_mlp"]:
                _vae_fuse_mlp[k] = sd["state_dict_vae_fuse_mlp"][k]
            self.vae_fuse_mlp.load_state_dict(_vae_fuse_mlp)

            _unet_fuse_mlp = self.unet_fuse_mlp.state_dict()
            for k in sd["state_dict_unet_fuse_mlp"]:
                _unet_fuse_mlp[k] = sd["state_dict_unet_fuse_mlp"][k]
            self.unet_fuse_mlp.load_state_dict(_unet_fuse_mlp)

            #self.W = nn.Parameter(sd["w"], requires_grad=False)

            embeddings_state_dict = sd["state_embeddings"]
            self.vae_block_embeddings.load_state_dict(embeddings_state_dict['state_dict_vae_block'])
            self.unet_block_embeddings.load_state_dict(embeddings_state_dict['state_dict_unet_block'])
        else:
            print("Initializing model with random weights")
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)

        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        self.vae_lora_layers = []
        for name, module in vae.named_modules():
            if 'base_layer' in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
                
        for name, module in vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])

        for name, module in unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_layer_dict = {name: get_layer_number(name) for name in self.unet_lora_layers}

        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

        size = 512
        edge_width = 50
        center_value = 1.0
        edge_value = 1.5

        # 创建一个全 1 的矩阵
        matrix = torch.full((size, size), center_value).cuda()

        # 创建渐变边界
        gradient = torch.linspace(center_value, edge_value, edge_width)

        # 设置平滑边缘区域值
        # 上边缘
        for i in range(edge_width):
            matrix[i, :] = gradient[i]

        # 下边缘
        for i in range(edge_width):
            matrix[-(i + 1), :] = gradient[i]

        # 左边缘
        for i in range(edge_width):
            matrix[:, i] = gradient[i]

        # 右边缘
        for i in range(edge_width):
            matrix[:, -(i + 1)] = gradient[i]
        self.opt_weight = matrix
        self.guidance_scale = 0.75

    def set_eval(self):
        self.W.eval()
        self.unet.eval()
        self.vae.eval()
        self.vae_de_mlp.eval()
        self.unet_de_mlp.eval()
        self.vae_block_mlp.eval()
        self.unet_block_mlp.eval()
        self.vae_fuse_mlp.eval()
        self.unet_fuse_mlp.eval()

        self.vae_block_embeddings.requires_grad_(False)
        self.unet_block_embeddings.requires_grad_(False)

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.W.train()
        self.unet.train()
        self.vae.train()
        self.vae_de_mlp.train()
        self.unet_de_mlp.train()
        self.vae_block_mlp.train()
        self.unet_block_mlp.train()
        self.vae_fuse_mlp.train()
        self.unet_fuse_mlp.train()    

        self.vae_block_embeddings.requires_grad_(True)
        self.unet_block_embeddings.requires_grad_(True)

        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

        self.unet.conv_in.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def forward_embedding(self,encoded_control , model_pred_pos,model_pred_neg,guidance_scale=0.75):
        
        x_denoised_pos = self.sched.step(model_pred_pos, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image_pos = (self.vae.decode(x_denoised_pos / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        x_denoised_neg = self.sched.step(model_pred_neg, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image_neg = (self.vae.decode(x_denoised_neg / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        model_pred = model_pred_neg + guidance_scale * (model_pred_pos - model_pred_neg)
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image_pos,output_image_neg,output_image

    def forward(self, c_t, deg_score, prompt_neg,prompt_pos=None,guidance_scale=0.75):
        neg_caption_tokens = self.tokenizer(prompt_neg, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
        neg_caption_enc = self.text_encoder(neg_caption_tokens)[0]
 
        if prompt_pos is not None:
            # encode the text prompt
            pos_caption_tokens = self.tokenizer(prompt_pos, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            pos_caption_enc = self.text_encoder(pos_caption_tokens)[0]
        
        B, P, P = deg_score.shape
        deg_score = deg_score.reshape(B,P*P)
        deg_proj = self.W(deg_score)

        # degradation mlp forward
        vae_de_c_embed = self.vae_de_mlp(deg_proj)
        unet_de_c_embed = self.unet_de_mlp(deg_proj)

        # block embedding mlp forward
        vae_block_c_embeds = self.vae_block_mlp(self.vae_block_embeddings.weight)
        unet_block_c_embeds = self.unet_block_mlp(self.unet_block_embeddings.weight)
        vae_embeds = self.vae_fuse_mlp(torch.cat([vae_de_c_embed.unsqueeze(1).repeat(1, vae_block_c_embeds.shape[0], 1), \
            vae_block_c_embeds.unsqueeze(0).repeat(vae_de_c_embed.shape[0],1,1)], -1))
        unet_embeds = self.unet_fuse_mlp(torch.cat([unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1), \
            unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.shape[0],1,1)], -1))

        for layer_name, module in self.vae.named_modules():
            if layer_name in self.vae_lora_layers:
                split_name = layer_name.split(".")
                if split_name[1] == 'down_blocks':
                    block_id = int(split_name[2])
                    vae_embed = vae_embeds[:, block_id]
                elif split_name[1] == 'mid_block':
                    vae_embed = vae_embeds[:, -2]
                else:
                    vae_embed = vae_embeds[:, -1]
                module.de_mod = vae_embed.reshape(-1, self.lora_rank_vae, self.lora_rank_vae)

        for layer_name, module in self.unet.named_modules():
            if layer_name in self.unet_lora_layers:
                split_name = layer_name.split(".")

                if split_name[0] == 'down_blocks':
                    block_id = int(split_name[1])
                    unet_embed = unet_embeds[:, block_id]
                elif split_name[0] == 'mid_block':
                    unet_embed = unet_embeds[:, 4]
                elif split_name[0] == 'up_blocks':
                    block_id = int(split_name[1]) + 5
                    unet_embed = unet_embeds[:, block_id]
                else:
                    unet_embed = unet_embeds[:, -1]
                module.de_mod = unet_embed.reshape(-1, self.lora_rank_unet, self.lora_rank_unet)

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        
        if prompt_pos is not None:
            model_pred_pos = self.unet(encoded_control, self.timesteps, encoder_hidden_states=pos_caption_enc,).sample
            # model_pred_neg = F.interpolate(model_pred_neg,(512,512), mode='nearest')
            # model_pred_pos = F.interpolate(model_pred_pos,(512,512), mode='nearest')
            #model_pred = model_pred_neg + self.opt_weight * (model_pred_pos - model_pred_neg)
            #model_pred= F.interpolate(model_pred,(64,64), mode='nearest')
        
        model_pred_neg = self.unet(encoded_control, self.timesteps, encoder_hidden_states=neg_caption_enc,).sample
        x_denoised_pos = self.sched.step(model_pred_pos, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image_pos = (self.vae.decode(x_denoised_pos / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        x_denoised_neg = self.sched.step(model_pred_neg, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image_neg = (self.vae.decode(x_denoised_neg / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        model_pred = model_pred_neg + guidance_scale * (model_pred_pos - model_pred_neg)
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image_pos,output_image_neg,output_image,model_pred_pos,model_pred_neg,encoded_control

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip_conv" in k}
        sd["state_dict_vae_de_mlp"] = {k: v for k, v in self.vae_de_mlp.state_dict().items()}
        sd["state_dict_unet_de_mlp"] = {k: v for k, v in self.unet_de_mlp.state_dict().items()}
        sd["state_dict_vae_block_mlp"] = {k: v for k, v in self.vae_block_mlp.state_dict().items()}
        sd["state_dict_unet_block_mlp"] = {k: v for k, v in self.unet_block_mlp.state_dict().items()}
        sd["state_dict_vae_fuse_mlp"] = {k: v for k, v in self.vae_fuse_mlp.state_dict().items()}
        sd["state_dict_unet_fuse_mlp"] = {k: v for k, v in self.unet_fuse_mlp.state_dict().items()}
        sd["state_dict_unet_fuse_W"] = {k: v for k, v in self.W.state_dict().items()}

        sd["state_embeddings"] = {
                    "state_dict_vae_block": self.vae_block_embeddings.state_dict(),
                    "state_dict_unet_block": self.unet_block_embeddings.state_dict(),
                }

        torch.save(sd, outf)
