from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import hf_hub_download
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import tempfile
import builtins

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import custom_bwd, custom_fwd


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_tokenizer_same_repo(repo_id: str):
    """
    Keep tokenizer content from the same repo for fair reproduction.
    """
    try:
        return CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer"), None
    except Exception as e:
        print(f'[WARN] direct tokenizer load failed for {repo_id}/tokenizer: {e}')
        print(f'[WARN] fallback to hf_hub_download with the same tokenizer files from {repo_id}')

        tmp_dir = Path(tempfile.mkdtemp(prefix="paintit_tokenizer_"))

        required_files = [
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]

        optional_files = [
            "tokenizer.json",
            "added_tokens.json",
        ]

        for fname in required_files:
            src = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                subfolder="tokenizer",
            )
            shutil.copy2(src, tmp_dir / fname)

        for fname in optional_files:
            try:
                src = hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    subfolder="tokenizer",
                )
                shutil.copy2(src, tmp_dir / fname)
            except Exception:
                pass

        tok = CLIPTokenizer.from_pretrained(str(tmp_dir), local_files_only=True)
        return tok, str(tmp_dir)


def load_text_encoder_same_repo(repo_id: str, device):
    """
    Keep text_encoder content from the same repo for fair reproduction.
    First try original path; if transformers URL assembly breaks under mirror,
    download the exact text_encoder files from the same repo and load locally.
    """
    try:
        enc = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder").to(device)
        return enc, None
    except Exception as e:
        print(f'[WARN] direct text_encoder load failed for {repo_id}/text_encoder: {e}')
        print(f'[WARN] fallback to hf_hub_download with the same text_encoder files from {repo_id}')

        tmp_dir = Path(tempfile.mkdtemp(prefix="paintit_text_encoder_"))

        # Keep the same model content. Use fp32 bin for closest reproduction.
        required_files = [
            "config.json",
            "pytorch_model.bin",
        ]

        for fname in required_files:
            src = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                subfolder="text_encoder",
            )
            shutil.copy2(src, tmp_dir / fname)

        enc = CLIPTextModel.from_pretrained(str(tmp_dir), local_files_only=True).to(device)
        return enc, str(tmp_dir)


class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.1', hf_key=None, min=0.02, max=0.98,
                 use_fp16=True, enable_attention_slicing=True, unet_chunk_size=2):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self._tokenizer_cache_dir = None
        self._text_encoder_cache_dir = None
        self.use_fp16 = bool(use_fp16 and self.device.type == 'cuda')
        self.unet_chunk_size = builtins.max(1, int(unet_chunk_size))

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "Manojb/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "Manojb/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer, self._tokenizer_cache_dir = load_tokenizer_same_repo(model_key)
        self.text_encoder, self._text_encoder_cache_dir = load_text_encoder_same_repo(model_key, self.device)
        unet_dtype = torch.float16 if self.use_fp16 else torch.float32
        self.unet = UNet2DConditionModel.from_pretrained(
            model_key, subfolder="unet", torch_dtype=unet_dtype
        ).to(self.device)

        if enable_attention_slicing and hasattr(self.unet, 'set_attention_slice'):
            self.unet.set_attention_slice('auto')

        if is_xformers_available():
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f'[WARN] xformers memory efficient attention unavailable: {e}')

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        # self.ts_sampler = TimestepSampler(self.scheduler, device=device)
        # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * min)
        self.max_step = int(self.num_train_timesteps * max)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f'[INFO] loaded stable diffusion!')


    def _run_unet(self, latents, timesteps, text_embeddings):
        unet_dtype = next(self.unet.parameters()).dtype
        latents = latents.to(device=self.device, dtype=unet_dtype)
        text_embeddings = text_embeddings.to(device=self.device, dtype=unet_dtype)
        if isinstance(timesteps, torch.Tensor):
            timesteps = timesteps.to(self.device)

        if self.device.type == 'cuda' and unet_dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type='cuda', dtype=unet_dtype):
                noise_pred = self.unet(latents, timesteps, encoder_hidden_states=text_embeddings).sample
        else:
            noise_pred = self.unet(latents, timesteps, encoder_hidden_states=text_embeddings).sample
        return noise_pred


    def _predict_noise_chunked(self, latents_noisy, timesteps, text_embeddings):
        # text_embeddings must be [2B, S, C], where first B is uncond and next B is cond.
        B = latents_noisy.shape[0]
        if text_embeddings.shape[0] != 2 * B:
            raise ValueError(f'Expected text embeddings shape[0] == 2*B ({2 * B}), got {text_embeddings.shape[0]}')

        chunk_size = min(self.unet_chunk_size, B)
        while True:
            try:
                noise_pred_chunks = []
                for start in range(0, B, chunk_size):
                    end = min(start + chunk_size, B)

                    latents_chunk = latents_noisy[start:end]
                    t_chunk = timesteps[start:end]
                    emb_chunk = torch.cat([
                        text_embeddings[start:end],
                        text_embeddings[B + start:B + end]
                    ], dim=0)

                    latent_model_input = torch.cat([latents_chunk, latents_chunk], dim=0)
                    tt = torch.cat([t_chunk, t_chunk], dim=0)
                    noise_pred_chunk = self._run_unet(latent_model_input, tt, emb_chunk)
                    noise_pred_chunks.append(noise_pred_chunk)

                return torch.cat(noise_pred_chunks, dim=0)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and chunk_size > 1:
                    next_chunk = max(1, chunk_size // 2)
                    print(f'[WARN] CUDA OOM in UNet chunk_size={chunk_size}, retrying with chunk_size={next_chunk}')
                    chunk_size = next_chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise


    def get_text_embeds(self, prompt, negative_prompt=[''], batch=1):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        B, S = text_embeddings.shape[:2]
        text_embeddings = text_embeddings.repeat(1, batch, 1).view(B * batch, S, -1)

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        B, S = uncond_embeddings.shape[:2]
        uncond_embeddings = uncond_embeddings.repeat(1, batch, 1).view(B * batch, S, -1)

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings


    def batch_train_step(self, text_embeddings, pred_rgb, guidance_scale=30, as_latent=False, min_step=0.02, max_step=0.98, phi=0.5, return_t=False):
        B = pred_rgb.shape[0]
        min_step = int(self.num_train_timesteps * min_step)
        max_step = int(self.num_train_timesteps * max_step)

        if as_latent:
            # directly downsample input as latent
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            if pred_rgb.shape[-2:] != (512, 512):
                # interp to 512x512 to be fed into vae.
                pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            else:
                pred_rgb_512 = pred_rgb
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device).repeat(B)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            noise_pred = self._predict_noise_chunked(latents_noisy, t, text_embeddings).to(dtype=latents.dtype)

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = w.view(-1, 1, 1, 1) * (noise_pred - noise)

        # clip grad for stable training?
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)

        if return_t:
            return loss, t
        else:
            return loss


    @torch.no_grad()
    def refine(self, text_embeddings, pred_rgb, guidance_scale=100, steps=50, strength=0.8):

        batch_size = pred_rgb.shape[0]
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512)
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self._run_unet(latent_model_input, t, text_embeddings).to(dtype=latents.dtype)

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]
        return imgs

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5,
                        latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                                  device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self._run_unet(latent_model_input, t, text_embeddings).to(dtype=latents.dtype)

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50,
                      guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,
                                       num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale)  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'],
                        help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
