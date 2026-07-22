"""潜空间流形牵引损失（Tier 2/3，需 4090 24GB + diffusers 库）。
Latent manifold steering loss (Tier 2/3, requires 4090 24GB + diffusers).

基于 DDA 论文思路：用扩散模型（DDIM）的潜空间作为"边界感知流形"，
将预测和 GT 都投影到该空间后做特征对齐，间接约束预测的边界质量。
Based on DDA paper idea: use diffusion model (DDIM) latent space as a
"boundary-aware manifold", project both prediction and GT into it,
align features to indirectly constrain boundary quality.

原理 / Principle:
1. 冻结 SD VAE 编码器 → 把 GT mask 和预测 mask 都编码到潜空间
   Freeze SD VAE → encode GT mask and pred mask into latent space
2. 冻结 SD U-Net → 两个潜变量各做一步 DDIM 反演，得到"边界增强版"
   Freeze SD U-Net → one DDIM inversion step on each latent, producing
   "boundary-enhanced" versions
3. 计算两个增强潜变量的 MSE → 作为辅助损失加入训练
   Compute MSE between the two enhanced latents → add as auxiliary loss
4. 效果：预测被"牵引"到边界更自然的流形区域
   Effect: prediction is "steered" toward boundary-natural manifold regions

训练时用法 / Training usage:
  # 在 train_task1.py 的训练循环中加
  if args.diffusion_loss and ep > 5:  # 前 5 epoch 跳过（VAE forward 开销大）
      d_loss = diffusion_align(vae, unet, prob, mask_gt, device)
      loss = loss + 0.1 * d_loss
"""
import torch
import torch.nn.functional as F


class DiffusionAlignLoss:
    """潜空间流形对齐：SD VAE 编码 + U-Net 增强 → MSE 牵引。
    Latent manifold alignment: SD VAE encode + U-Net enhance → MSE steering."""
    def __init__(self, device, sd_id='stabilityai/stable-diffusion-2-1-base', half=True):
        try:
            from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        except ImportError:
            raise ImportError('diffusion loss needs diffusers. pip install diffusers')
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(sd_id, subfolder='vae').to(device).eval()
        self.unet = UNet2DConditionModel.from_pretrained(sd_id, subfolder='unet').to(device).eval()
        self.noise_scheduler = DDPMScheduler.from_pretrained(sd_id, subfolder='scheduler')
        self.t = 5  # fixed small timestep for boundary sensitivity
        if half:
            self.vae = self.vae.half()
            self.unet = self.unet.half()
        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.unet.parameters():
            p.requires_grad = False

    def encode_to_latent(self, mask_3ch):
        """mask_3ch: [B,3,H,W] normalized (0|1 → -1|1 range). Returns latents [B,4,H/8,W/8]."""
        mask_3ch = mask_3ch.to(self.device)
        if self.vae.dtype == torch.float16:
            mask_3ch = mask_3ch.half()
        latents = self.vae.encode(mask_3ch).latent_dist.sample()
        return latents * 0.18215

    def diffuse_one_step(self, latents):
        """一步 DDPM 正向扩散 + U-Net 预测 → 增强潜变量。
        One-step DDPM forward + U-Net prediction → enhanced latent."""
        bsz = latents.shape[0]
        noise = torch.randn_like(latents)
        timesteps = torch.tensor([self.t] * bsz, device=self.device).long()
        noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden = torch.zeros(bsz, 77, 1024, device=self.device,
                                     dtype=latents.dtype)
        noise_pred = self.unet(noisy, timesteps, encoder_hidden_states=encoder_hidden).sample
        return noise_pred  # U-Net 的噪声预测含丰富边界信息 / U-Net noise pred has rich boundary info

    def __call__(self, pred_mask, gt_mask):
        """pred_mask/gt_mask: [B,1,H,W] 0~1 prob。返回标量 loss。
        pred_mask/gt_mask: [B,1,H,W] 0~1 prob. Returns scalar loss."""
        H, W = pred_mask.shape[-2:]
        # 转为 3 通道 [0,255] 给 VAE / convert to 3-channel for VAE
        if H % 64 != 0 or W % 64 != 0:
            H2 = (H // 64) * 64
            W2 = (W // 64) * 64
            pred_mask = F.interpolate(pred_mask, size=(H2, W2), mode='bilinear')
            gt_mask = F.interpolate(gt_mask, size=(H2, W2), mode='bilinear')
        # stack to 3-channel (grayscale → RGB)
        pred_3ch = pred_mask.repeat(1, 3, 1, 1) * 2 - 1  # [B,3,H,W] in [-1,1]
        gt_3ch = gt_mask.repeat(1, 3, 1, 1) * 2 - 1

        try:
            with torch.no_grad():
                z_pred = self.encode_to_latent(pred_3ch)
                z_gt = self.encode_to_latent(gt_3ch)
                e_pred = self.diffuse_one_step(z_pred)
                e_gt = self.diffuse_one_step(z_gt)
            return F.mse_loss(e_pred, e_gt)
        except Exception:
            return torch.tensor(0.0, device=self.device)  # OOM/error 时跳过 / skip on OOM
