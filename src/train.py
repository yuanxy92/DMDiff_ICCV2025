import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

import gc
import lpips
import clip
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from s3diff_super import OPT_S3Diff as S3Diff
from my_utils.training_utils import parse_args_paired_training, PairedDataset, degradation_proc
import pyiqa


def clip_and_quantize(scores, clip_range=(0, 80), quantization_step=5):
    """
    Clip MUSIQ scores to a range and quantize them into levels using PyTorch.
    
    Args:
        scores (torch.Tensor): Input MUSIQ scores of shape (B, P, P).
        clip_range (tuple): Min and max values for clipping (inclusive).
        quantization_step (int): Step size for quantization.
    
    Returns:
        torch.Tensor: Quantized scores of shape (B, P, P), as float type.
    """
    min_val, max_val = clip_range
    
    # Step 1: Clip the scores
    clipped_scores = torch.clamp(scores, min=min_val, max=max_val)
    
    # Step 2: Quantize the scores (divide by step, round, then multiply by step)
    quantized_scores = torch.round(clipped_scores / quantization_step)
    return quantized_scores.float()

def generate_quality_matrix(size=7, low_value=0.75, high_value=1.0):
    # Create a 2D Gaussian-like distribution centered at the middle
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Generate a Gaussian-like distribution with center value high, and edges low
    matrix = np.exp(-(X**2 + Y**2) * 2)
    
    # Normalize to the range [low_value, high_value]
    matrix = low_value + (high_value - low_value) * matrix
    
    return torch.tensor(matrix, dtype=torch.float32).cuda()[None,...]

def calculate_iqa_batch(images, iqa_metric, patch_size=100, stride=50):
    """
    Calculate IQA scores for a batch of images using patch-based evaluation.
    
    Args:
        images (torch.Tensor): Input images of shape (B, 3, H, W), normalized to [0, 1].
        iqa_metric: IQA metric model.
        patch_size (int): Size of each patch.

    Returns:
        torch.Tensor: IQA scores for each patch, reshaped to (B, P, P), where P = H // patch_size.
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch size."
    
    P = H // patch_size  # Number of patches per dimension
    
    # Divide images into patches
    patches = images.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # Shape: (B, P, P, C, patch_size, patch_size)
    patches = patches.reshape(-1, C, patch_size, patch_size)  # Shape: (B * P * P, C, patch_size, patch_size)
    
    # Compute IQA scores for all patches
    iqa_scores = iqa_metric(patches)  # Shape: (B * P * P)
    
    # Reshape scores back to (B, P, P)
    P = 7
    iqa_scores = iqa_scores.view(B, P, P)
    
    iqa_scores = clip_and_quantize(iqa_scores)
    #iqa_scores =  iqa_scores.reshape(B,P*P)
    opt_scores =generate_quality_matrix(P)
    iqa_scores *=  opt_scores

    return iqa_scores

def main(args):

    # init and save configs
    config = OmegaConf.load(args.base_config)

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path
    #kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=False)]
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # initialize degradation estimation network
    net_de = pyiqa.create_metric('musiq', device='cuda')

    # initialize net_sr
    net_sr = S3Diff(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae, sd_path=sd_path, pretrained_path=args.pretrained_path)
    net_sr.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_sr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_sr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    layers_to_opt = layers_to_opt + list(net_sr.vae_block_embeddings.parameters()) + list(net_sr.unet_block_embeddings.parameters())
    layers_to_opt = layers_to_opt + list(net_sr.vae_de_mlp.parameters()) + list(net_sr.unet_de_mlp.parameters()) + \
        list(net_sr.vae_block_mlp.parameters()) + list(net_sr.unet_block_mlp.parameters()) + \
        list(net_sr.vae_fuse_mlp.parameters()) + list(net_sr.unet_fuse_mlp.parameters()) + list(net_sr.W.parameters())

    for n, _p in net_sr.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_sr.unet.conv_in.parameters())

    for n, _p in net_sr.vae.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)

    dataset_train = PairedDataset(config.train)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = PairedDataset(config.validation)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)


    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    net_sr, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_sr, optimizer, dl_train, lr_scheduler
    )

    net_de, net_lpips = accelerator.prepare(net_de, net_lpips)
    # # renorm with image net statistics
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_sr.to(accelerator.device, dtype=weight_dtype)
    net_de.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    global_step = 0
    args.neg_prob = 0.1
    super_prob = 0.7
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            if global_step < 5:
                args.neg_prob = 1
            else:
                args.neg_prob = 0.1
            if global_step < 500:
                super_prob = 0.75
            if global_step < 1000:
                super_prob = 0.6
            if global_step < 1500:
                super_prob = 0.5
            if global_step < 2000:
                super_prob = 0.4
            if global_step > 2000:
                super_prob = 0.3
            

            l_acc = [net_sr]
            with accelerator.accumulate(*l_acc):
                x_src, x_tgt, x_ori_size_src,x_tgt2 = degradation_proc(config, batch, accelerator.device)

                B, C, H, W = x_src.shape
                with torch.no_grad():
                    deg_score = calculate_iqa_batch(x_ori_size_src.detach(),net_de).detach()

                batch_prob = torch.rand(1, device=accelerator.device)  # 放在正确设备上
                use_neg_prompt = batch_prob < args.neg_prob  
                neg_tag_prompt = [args.neg_prompt for _ in range(B)]
                
                if use_neg_prompt:
                    # 整个 batch 使用负向提示
                    mixed_tgt = x_src  # 对应负向目标
                    pos_tag_prompt = None
                    x_src = x_tgt
                else:
                    # 整个 batch 使用正向提示
                    pos_tag_prompt = [args.pos_prompt for _ in range(B)]
                    mixed_tgt = x_tgt  # 对应正向目标
                    if batch_prob < super_prob:
                        super_tag_prompt = ['A well-balanced image with acceptable sharpness, average detail, and natural colors, presenting the scene in a simple and straightforward manner' for _ in range(B)]
                        # with torch.no_grad():
                        #     x_src = net_sr(x_src.detach(), deg_score, neg_tag_prompt,pos_tag_prompt)
                        pos_tag_prompt = super_tag_prompt
                        mixed_tgt = x_tgt2

                
                x_tgt_pred = net_sr(x_src.detach(), deg_score, neg_tag_prompt,pos_tag_prompt)
                loss_l2 = F.mse_loss(x_tgt_pred.float(), mixed_tgt.detach().float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), mixed_tgt.detach().float()).mean() * args.lambda_lpips

                loss = loss_l2 + loss_lpips

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_sr).save_model(outf)

                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    if global_step % args.eval_freq == 1:
                        l_l2, l_lpips = [], []
        
                        val_count = 0
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src, x_tgt, x_ori_size_src,x_tgt2 = degradation_proc(config, batch_val, accelerator.device)
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                with torch.no_grad():
                                    deg_score = calculate_iqa_batch(x_ori_size_src.detach(),net_de).detach()

                                    pos_tag_prompt = [args.pos_prompt for _ in range(B)]
                                    neg_tag_prompt = [args.neg_prompt for _ in range(B)]
                                    super_tag_prompt = ['A well-balanced image with acceptable sharpness, average detail, and natural colors, presenting the scene in a simple and straightforward manner' for _ in range(B)]
                                    x_tgt_pred_neg = accelerator.unwrap_model(net_sr)(x_tgt.detach(), deg_score, neg_tag_prompt,None)
                                    x_tgt_pred = accelerator.unwrap_model(net_sr)(x_src.detach(), deg_score, neg_tag_prompt,pos_tag_prompt)
                                    x_tgt_super = accelerator.unwrap_model(net_sr)(x_src.detach(), deg_score, neg_tag_prompt,super_tag_prompt)
                                
                                # compute the reconstruction losses
                                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float(), reduction="mean")
                                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.detach().float()).mean()

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())

                            if args.save_val and val_count < 5:
                                x_src = x_src.cpu().detach() * 0.5 + 0.5
                                x_tgt = x_tgt.cpu().detach() * 0.5 + 0.5
                                x_tgt2 = x_tgt2.cpu().detach() * 0.5 + 0.5
                                x_tgt_pred = x_tgt_pred.cpu().detach() * 0.5 + 0.5
                                x_tgt_pred_neg = x_tgt_pred_neg.cpu().detach() * 0.5 + 0.5
  
                                x_tgt_super = x_tgt_super.cpu().detach() * 0.5 + 0.5
                                combined = torch.cat([x_src, x_tgt_pred_neg, x_tgt_pred,x_tgt,x_tgt_super,x_tgt2], dim=3)
                                output_pil = transforms.ToPILImage()(combined[0])
                                outf = os.path.join(args.output_dir, f"val_{step}.png")
                                output_pil.save(outf)
                                val_count += 1

                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
