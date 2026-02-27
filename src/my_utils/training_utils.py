import argparse
import json
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from glob import glob

import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan")
    parser.add_argument("--gan_loss_type", default="multilevel_hinge")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5.0, type=float)
    parser.add_argument("--lambda_l2", default=2.0, type=float)
    parser.add_argument("--base_config", default="./configs/sr.yaml", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_val", default=True, action="store_false")
    parser.add_argument("--num_samples_eval", type=int, default=5, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")

    # details about the model architecture
    parser.add_argument("--sd_path")
    parser.add_argument("--pretrained_path", type=str, default=None,)
    parser.add_argument("--embedding_path", type=str, default=None,)
    parser.add_argument("--gt_path", type=str, default=None,)
    parser.add_argument("--de_net_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=16, type=int)
    parser.add_argument("--neg_prob", default=0.05, type=float)
    parser.add_argument("--pos_prompt", type=str, default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")
    parser.add_argument("--neg_prompt", type=str, default="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth")

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=100000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "piecewise_constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=0.1, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def add_combined_noise_torch(
    image,
    pseudo_mask,
    stddev=0.1,
    scale=0.3,
    sparsity=0.5,
    gan=0.4,
    poisson_weight=0.4,
    poisson_L=30,
    brightness_sensitive=True
):
    """
    对 torch.Tensor 图像添加组合噪声（高斯 + 泊松 + 稀疏 + 亮度调节）

    Args:
        image (Tensor): [C,H,W] 或 [B,C,H,W]，值范围 [0,1]
        stddev (float): 高斯噪声标准差
        scale (float): 控制块状程度
        sparsity (float): 0~1，控制稀疏性
        gan (float): 高斯噪声强度
        poisson_weight (float): 泊松噪声强度
        poisson_L (float): 泊松噪声放大系数（越小噪声越大）
        brightness_sensitive (bool): 是否基于亮度调整噪声强度
    Returns:
        Tensor: 噪声后的图像，值范围 [0,1]
    """

    is_batched = image.dim() == 4
    if not is_batched:
        image = image.unsqueeze(0)

    B, C, H, W = image.shape
    h_small, w_small = max(1, int(H * scale)), max(1, int(W * scale))

    # === 亮度感知掩码 ===
    if brightness_sensitive:
        with torch.no_grad():
            gray = image.mean(dim=1, keepdim=True)  # [B,1,H,W]
            brightness_factor = 1.0 - gray  # 暗处值大，亮处值小
            brightness_factor = torch.nn.functional.interpolate(brightness_factor, size=(h_small, w_small), mode='bilinear', align_corners=False)
    else:
        brightness_factor = torch.ones((B, 1, h_small, w_small), device=image.device)

    # === 高斯噪声 ===
    noise = torch.randn((B, C, h_small, w_small), device=image.device) * stddev
    if sparsity is not None:
        mask = (torch.rand((B, 1, h_small, w_small), device=image.device) < sparsity).float()
        noise *= mask
    noise *= brightness_factor
    noise = torch.nn.functional.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)
    gauss_noisy = gan * noise

    # === 泊松噪声 ===
    poisson_input = torch.clamp(image * poisson_L, min=0)
    poisson_noise = (torch.poisson(poisson_input) / poisson_L) - image
    if brightness_sensitive:
        poisson_noise *= torch.nn.functional.interpolate(brightness_factor, size=(H, W), mode='bilinear', align_corners=False)

    

    poisson_noisy = poisson_weight * poisson_noise
    total_noise = gauss_noisy + poisson_noisy

    if pseudo_mask is not None:
        pseudo_mask = pseudo_mask.view(B, 1, 1, 1).float()  # [B,1,1,1]
        total_noise = total_noise * pseudo_mask

    # === 合成并裁剪 ===
    noisy_image = image + total_noise
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)

    return noisy_image if is_batched else noisy_image.squeeze(0)

# @DATASET_REGISTRY.register(suffix='basicsr')
class PairedDataset(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(PairedDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        if 'crop_size' in opt:
            self.crop_size = opt['crop_size']
        else:
            self.crop_size = 512
        if 'image_type' not in opt:
            opt['image_type'] = 'png'

        # support multiple type of data: file path and meta data, remove support of lmdb
        self.paths = []
        self.paths2 = []
        self.lr_paths = []
        if 'meta_info' in opt:
            with open(self.opt['meta_info']) as fin:
                    paths = [line.strip().split(' ')[0] for line in fin]
                    self.paths = [v for v in paths]
            if 'meta_num' in opt:
                self.paths = sorted(self.paths)[:opt['meta_num']]
        if 'gt_path' in opt:
            if isinstance(opt['gt_path'], str):
                # Use rglob to recursively search for images
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path']).rglob('*.' + opt['image_type'])]))
            else:
                for path in opt['gt_path']:
                    self.paths.extend(sorted([str(x) for x in Path(path).rglob('*.' + opt['image_type'])]))
            import os
            gt_path2 = opt['gt_path'].replace('gt','gt_blur') if 'train' in os.path.basename(os.path.normpath(opt['gt_path'])) else opt['gt_path']
            self.paths2.extend(sorted([str(x) for x in Path(gt_path2).rglob('*.' + opt['image_type'])]))
        
        if 'lr_path' in opt:
            if isinstance(opt['lr_path'], str):
                # Use rglob to recursively search for images
                self.lr_paths.extend(sorted([str(x) for x in Path(opt['lr_path']).rglob('*.' + opt['image_type'])]))
            else:
                for path in opt['lr_path']:
                    self.lr_paths.extend(sorted([str(x) for x in Path(path).rglob('*.' + opt['image_type'])]))
                
        
        if 'num_pic' in opt:
            if 'val' or 'test' in opt:
                #random.shuffle(self.paths)
                self.paths = self.paths[:opt['num_pic']]
                self.lr_paths = self.lr_paths[:opt['num_pic']]
            else:
                self.paths = self.paths[:opt['num_pic']]
                self.lr_paths = self.lr_paths[:opt['num_pic']]

        if 'mul_num' in opt:
            self.paths = self.paths * opt['mul_num']
            self.lr_paths = self.lr_paths * opt['mul_num']
            # print('>>>>>>>>>>>>>>>>>>>>>')
            # print(self.paths)

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1
        # print(self.paths[:10])
        # print(self.lr_paths[:10])
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        lr_path = self.lr_paths[index]
        gt_path2 = self.paths2[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_bytes2 = self.file_client.get(gt_path2, 'gt2')
                lr_img_bytes = self.file_client.get(lr_path, 'lr')
            except (IOError, OSError) as e:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__()-1)
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)
        img_gt2 = imfrombytes(img_bytes2, float32=True)
        img_lr = imfrombytes(lr_img_bytes, float32=True)
        # filter the dataset and remove images with too low quality
        img_size = os.path.getsize(gt_path)
        img_size = img_size / 1024


        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_gt2 = img2tensor([img_gt2], bgr2rgb=True, float32=True)[0]
        img_lr = img2tensor([img_lr], bgr2rgb=True, float32=True)[0]
        if 'pseudo' in Path(lr_path).resolve().name.lower() and random.random() < 0.6:
            img_lr = add_combined_noise_torch(
            img_lr,
            None,
            stddev=random.uniform(0.1, 0.4),
            gan=random.uniform(0.1, 0.5),
            poisson_weight=random.uniform(0.1, 0.4),
            sparsity=random.uniform(0.2, 0.5),
            scale=random.uniform(0.2, 0.5),
            brightness_sensitive=True
        )
        
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'gt2': img_gt2, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path,'lr': img_lr,'lr_path': lr_path}
        return return_d

    def __len__(self):
        return len(self.paths)




def randn_cropinput(lq, gt, base_size=[64, 128, 256, 512]):
    cur_size_h = random.choice(base_size)
    cur_size_w = random.choice(base_size)
    init_h = lq.size(-2)//2
    init_w = lq.size(-1)//2
    lq = lq[:, :, init_h-cur_size_h//2:init_h+cur_size_h//2, init_w-cur_size_w//2:init_w+cur_size_w//2]
    gt = gt[:, :, init_h-cur_size_h//2:init_h+cur_size_h//2, init_w-cur_size_w//2:init_w+cur_size_w//2]
    assert lq.size(-1)>=64
    assert lq.size(-2)>=64
    return [lq, gt]




def degradation_proc(configs, batch, device, val=False, use_usm=False, resize_lq=True, random_size=False):

    """Degradation pipeline, modified from Real-ESRGAN:
    https://github.com/xinntao/Real-ESRGAN
    """

    jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
    usm_sharpener = USMSharp().cuda()  # do usm sharpening

    im_gt = batch['gt'].cuda()
    im_gt2 = batch['gt2'].cuda()
    im_lr = batch['lr'].cuda()
    
    # clamp and round
    #im_lq = torch.clamp(out, 0, 1.0)
    im_lq = torch.clamp(im_lr, 0, 1.0)
    ori_lq = im_lq
    '''
    mode = 'bilinear'
    ori_lq = F.interpolate(
        ori_lq,
        size=(400,400),
        mode=mode,
    )
    '''
    # random crop
    gt_size = configs.degradation['gt_size']
    #mode = random.choice(['area', 'bilinear', 'bicubic'])
    mode = 'bilinear'
    im_lq = F.interpolate(
            im_lq,
            size=(128,128),
            mode=mode,
            )
    
    im_gts, im_lq = paired_random_crop([im_gt,im_gt2], im_lq, gt_size, configs.sf)
    lq, gt ,gt2 = im_lq, im_gts[0], im_gts[1]
    

    if resize_lq:
        lq = F.interpolate(
                lq,
                size=(gt.size(-2),
                      gt.size(-1)),
                mode='bicubic',
                )


    # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
    lq = lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
    lq = lq * 2 - 1.0 # TODO 0~1?
    gt = gt * 2 - 1.0
    gt2 = gt2 * 2 - 1.0

    # if random_size:
    #     lq, gt = randn_cropinput(lq, gt)

    lq = torch.clamp(lq, -1.0, 1.0)

    return lq.to(device), gt.to(device), ori_lq.to(device),gt2.to(device)

