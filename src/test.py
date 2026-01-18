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

from s3diff_super_test import OPT_S3Diff as S3Diff
from my_utils.testing_utils import parse_args_paired_training, PairedDataset, degradation_proc
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
    net_sr.set_eval()

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

    dataset_test = PairedDataset(config.validation)
    dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)


    net_sr, net_de, net_lpips = accelerator.prepare(net_sr, net_de, net_lpips)
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

    progress_bar = tqdm(range(0, len(dataset_test)), initial=0, desc="Steps")


    # start the training loop
    args.neg_prob = 0.
                    
    val_count = 0

    save_embedding = True
    for step, batch_val in tqdm(enumerate(dl_test)):
        x_src, x_tgt, x_ori_size_src = degradation_proc(config, batch_val, accelerator.device)
        B, C, H, W = x_src.shape
        assert B == 1, "Use batch size 1 for eval."
        img_name = os.path.basename(batch_val['lr_path'][0])

        with torch.no_grad():
            # forward pass
            with torch.no_grad():
                deg_score = calculate_iqa_batch(x_ori_size_src.detach(),net_de).detach()
            args.neg_prompt = 'A well-balanced image with acceptable sharpness, average detail, and natural colors, presenting the scene in a simple and straightforward manner'
            pos_tag_prompt = [args.pos_prompt for _ in range(B)]
            neg_tag_prompt = [args.neg_prompt for _ in range(B)]
            x_tgt_pred_pos,x_tgt_pred_neg,x_tgt_pred,model_pred_pos,model_pred_neg,encoded_control = accelerator.unwrap_model(net_sr)(x_src.detach(), deg_score, neg_tag_prompt,pos_tag_prompt,args.inject_scale)
       
            if save_embedding:
                embeddingdir = os.path.join(args.output_dir, 'embedding')
                os.makedirs(embeddingdir,exist_ok=True)
                pt_name = img_name.replace('png','pt')
                savepath = os.path.join(embeddingdir,pt_name)
                torch.save({"neg": model_pred_neg, "pos": model_pred_pos,'encoded_control':encoded_control}, savepath)
          
            
            x_tgt_pred = x_tgt_pred.cpu().detach() * 0.5 + 0.5
            x_tgt_pred_neg = x_tgt_pred_neg.cpu().detach() * 0.5 + 0.5
            x_tgt_pred_pos = x_tgt_pred_pos.cpu().detach() * 0.5 + 0.5
            

            os.makedirs(os.path.join(args.output_dir, 'med'),exist_ok=True)
            outf = os.path.join(args.output_dir, 'med', img_name)
            output_pil = transforms.ToPILImage()(x_tgt_pred_neg[0])
            output_pil.save(outf)

            os.makedirs(os.path.join(args.output_dir, 'pos'),exist_ok=True)
            outf = os.path.join(args.output_dir, 'pos', img_name)
            output_pil = transforms.ToPILImage()(x_tgt_pred_pos[0])
            output_pil.save(outf)

            os.makedirs(os.path.join(args.output_dir, 'mix'),exist_ok=True)
            outf = os.path.join(args.output_dir, 'mix', img_name)
            output_pil = transforms.ToPILImage()(x_tgt_pred[0])
            output_pil.save(outf)
            val_count += 1
            progress_bar.update(1)

    
    gc.collect()
    torch.cuda.empty_cache()



if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
