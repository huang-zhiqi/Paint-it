import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import copy
import csv
import json
import math
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

from dc_pbr import skip
from nvdiff_render.material import *
from nvdiff_render.mesh import *
from nvdiff_render.obj import *
from nvdiff_render.render import *
from nvdiff_render.texture import *
from sd import StableDiffusion
from utils import *

OBJECT_PATH = "./data"
_GLCTX = None


def get_glctx():
    global _GLCTX
    if _GLCTX is None:
        if torch.cuda.is_available():
            _GLCTX = dr.RasterizeCudaContext()
        else:
            _GLCTX = dr.RasterizeGLContext()
    return _GLCTX


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)

    # model
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--lr_plateau", action="store_true")
    parser.add_argument("--decay_step", type=int, default=100)

    # training
    parser.add_argument("--sd_max_grad_norm", type=float, default=10.0)
    parser.add_argument("--n_iter", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--sd_min", type=float, default=0.2)
    parser.add_argument("--sd_max", type=float, default=0.98)
    parser.add_argument("--sd_min_l", type=float, default=0.2)
    parser.add_argument("--sd_min_r", type=float, default=0.3)
    parser.add_argument("--sd_max_l", type=float, default=0.5)
    parser.add_argument("--sd_max_r", type=float, default=0.98)
    parser.add_argument("--bg", type=float, default=0.25)
    parser.add_argument("--logging", type=str, default="True")
    parser.add_argument("--n_view", type=int, default=4)
    parser.add_argument("--env_scale", type=float, default=2.0)
    parser.add_argument("--envmap", type=str, default="data/irrmaps/mud_road_puresky_4k.hdr")
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--gd_scale", type=int, default=100)
    parser.add_argument("--uv_res", type=int, default=512)
    parser.add_argument("--final_render_chunk", type=int, default=8)
    parser.add_argument("--render_final_views", type=str, default="False")

    # export / normalization
    parser.add_argument("--texture_name", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mesh_norm_mode", type=str, default="texgaussian", choices=["paintit", "texgaussian"])
    parser.add_argument("--keep_only_required", type=str, default="True")

    # batch (TexGaussian-style)
    parser.add_argument("--tsv_path", type=str, default=None)
    parser.add_argument("--batch_path", type=str, default=None)
    parser.add_argument("--result_tsv", type=str, default=None)
    parser.add_argument("--caption_field", type=str, default="caption_long")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--auto_skip", type=str, default="True")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--workers_per_gpu", type=str, default="auto")

    args = parser.parse_args()

    args.logging = str2bool(args.logging)
    args.render_final_views = str2bool(args.render_final_views)
    args.keep_only_required = str2bool(args.keep_only_required)
    args.auto_skip = str2bool(args.auto_skip)

    if args.tsv_path is None and args.batch_path is not None:
        args.tsv_path = args.batch_path

    args.kd_min = [0.0, 0.0, 0.0, 0.0]
    args.kd_max = [1.0, 1.0, 1.0, 1.0]
    args.ks_min = [0.0, 0.08, 0.0]
    args.ks_max = [1.0, 1.0, 1.0]
    args.nrm_min = [-0.1, -0.1, 0.0]
    args.nrm_max = [0.1, 0.1, 1.0]

    if args.tsv_path is None:
        if not args.mesh_path:
            raise ValueError("--mesh_path is required in single-sample mode.")
        if not args.prompt:
            raise ValueError("--prompt is required in single-sample mode.")

    return args


def seed_all(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_model(args):
    input_depth = 3
    net = skip(
        input_depth,
        9,
        num_channels_down=[128] * 5,
        num_channels_up=[128] * 5,
        num_channels_skip=[128] * 5,
        filter_size_up=3,
        filter_size_down=3,
        upsample_mode="nearest",
        filter_skip_size=1,
        need_sigmoid=True,
        need_bias=True,
        pad="reflection",
        act_fun="LeakyReLU",
    ).type(torch.cuda.FloatTensor)

    params = list(net.parameters())
    lgt = light.load_env(args.envmap, scale=args.env_scale)
    for p in lgt.parameters():
        p.requires_grad = False

    optim = torch.optim.Adam(params, args.learning_rate, weight_decay=args.decay)
    activate_scheduler = args.lr_decay < 1 and args.decay_step > 0 and not args.lr_plateau
    lr_scheduler = None
    if activate_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.decay_step, gamma=args.lr_decay)

    return net, lgt, optim, lr_scheduler


def report_process(i, loss, exp_name):
    full_loss = 0.0
    log_message = f"[{exp_name}] iter: {i} "
    for loss_type, loss_val in loss.items():
        full_loss += loss_val
        log_message += f"{loss_type}: {loss_val:.3f} "
    loss["L_all"] = full_loss
    print(log_message)


def get_template_normal(h=512, w=512):
    return torch.cat(
        [
            torch.zeros((h, w, 1), device=device),
            torch.zeros((h, w, 1), device=device),
            torch.ones((h, w, 1), device=device),
        ],
        dim=-1,
    )[None, ...]


def compute_sd_step(min_val, max_val, iter_frac):
    return max_val - (max_val - min_val) * math.sqrt(iter_frac)


def normalize_mesh_texgaussian(mesh):
    with torch.no_grad():
        v = mesh.v_pos
        vmin = torch.min(v, dim=0).values
        vmax = torch.max(v, dim=0).values
        center = (vmax + vmin) * 0.5
        v = v - center[None, ...]
        distances = torch.linalg.norm(v, dim=-1)
        max_dist = torch.max(distances).item()
        if max_dist > 1e-8:
            v = v / max_dist
    return Mesh(v_pos=v, base=mesh)


def apply_mesh_normalization(mesh, mode):
    if mode == "texgaussian":
        return normalize_mesh_texgaussian(mesh)
    return unit_size(mesh)


def _texture_base(tex):
    data = tex.data[0] if isinstance(tex.data, list) else tex.data
    if data.ndim == 4:
        data = data[0]
    return data


def cleanup_sample_output(sample_dir):
    keep_files = {"mesh.obj", "albedo.png", "metallic.png", "roughness.png", "normal.png"}
    if not os.path.isdir(sample_dir):
        return
    for name in os.listdir(sample_dir):
        path = os.path.join(sample_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            continue
        if name not in keep_files:
            try:
                os.remove(path)
            except OSError:
                pass


def export_compact_assets(output_dir, vis_mesh, final_material, keep_only_required=True):
    os.makedirs(output_dir, exist_ok=True)

    write_obj(output_dir, vis_mesh, save_material=False)

    kd = _texture_base(final_material["kd"])[..., :3].detach().clamp(0, 1).cpu().numpy()
    util.save_image(os.path.join(output_dir, "albedo.png"), kd)

    ks = _texture_base(final_material["ks"]).detach().clamp(0, 1)
    if ks.shape[-1] < 3:
        raise RuntimeError("Unexpected ks channel count, cannot export metallic/roughness.")
    roughness = ks[..., 1].cpu().numpy()
    metallic = ks[..., 2].cpu().numpy()
    util.save_image(os.path.join(output_dir, "roughness.png"), roughness)
    util.save_image(os.path.join(output_dir, "metallic.png"), metallic)

    if "normal" in final_material.keys():
        normal = _texture_base(final_material["normal"])[..., :3]
        normal = (util.safe_normalize(normal) + 1.0) * 0.5
        normal = normal.detach().clamp(0, 1).cpu().numpy()
        util.save_image(os.path.join(output_dir, "normal.png"), normal)

    if keep_only_required:
        cleanup_sample_output(output_dir)


def render_final_views(args, vis_mesh, lgt, output_dir):
    circle_n_view = 120
    final_render_chunk = max(1, args.final_render_chunk)
    glctx = get_glctx()

    for elev in [-np.pi / 4, 0.0]:
        final_cam = sample_circle_view(n_view=circle_n_view, elev=elev, cam_radius=3.25)
        if elev == 0.0:
            out_view_dir = os.path.join(output_dir, "view_front")
        else:
            out_view_dir = os.path.join(output_dir, "view_top")
        os.makedirs(out_view_dir, exist_ok=True)

        for start_idx in range(0, circle_n_view, final_render_chunk):
            end_idx = min(start_idx + final_render_chunk, circle_n_view)
            final_buffers = render_mesh(
                glctx,
                vis_mesh,
                final_cam["mvp"][start_idx:end_idx],
                final_cam["campos"][start_idx:end_idx],
                lgt,
                final_cam["resolution"],
                spp=final_cam["spp"],
                msaa=True,
                background=None,
                bsdf="pbr",
            )

            final_obj_rgb = final_buffers["shaded"][..., 0:3].permute(0, 3, 1, 2).contiguous()
            final_obj_ws = final_buffers["shaded"][..., 3].unsqueeze(1)
            vis_mesh_img = final_obj_rgb * final_obj_ws + (1 - final_obj_ws) * 1

            for local_idx in range(end_idx - start_idx):
                global_idx = start_idx + local_idx
                if global_idx == 0:
                    if elev == 0.0:
                        torchvision.utils.save_image(final_obj_rgb[local_idx], os.path.join(output_dir, "final_front.png"))
                    else:
                        torchvision.utils.save_image(final_obj_rgb[local_idx], os.path.join(output_dir, "final_top.png"))
                torchvision.utils.save_image(vis_mesh_img[local_idx], os.path.join(out_view_dir, f"{global_idx:04}.png"))


def run_single_asset(args, guidance, mesh_path, prompt, sample_output_dir, sample_name):
    Path(sample_output_dir).mkdir(parents=True, exist_ok=True)

    sd_prompt = ", ".join((f"a DSLR photo of {prompt}", "best quality, high quality, extremely detailed, good geometry"))
    obj_f_uv, obj_v_uv, obj_f, obj_v = load_obj_uv(obj_path=mesh_path, device=device)

    mesh_t = Mesh(obj_v, obj_f, v_tex=obj_v_uv, t_tex_idx=obj_f_uv)
    mesh_t = apply_mesh_normalization(mesh_t, args.mesh_norm_mode)
    mesh_t = auto_normals(mesh_t)
    mesh_t = compute_tangents(mesh_t)

    input_uv_ = torch.randn((3, args.uv_res, args.uv_res), device=device)
    input_uv = (input_uv_ - torch.mean(input_uv_, dim=(1, 2)).reshape(-1, 1, 1)) / torch.std(input_uv_, dim=(1, 2)).reshape(-1, 1, 1)
    network_input = copy.deepcopy(input_uv.unsqueeze(0))

    net, lgt, optim, lr_scheduler = get_model(args)

    neg_prompt = "deformed, extra digit, fewer digits, cropped, worst quality, low quality, smoke"
    text_z = []
    for d in ["front", "side", "back", "overhead"]:
        text_z.append(guidance.get_text_embeds([f"{sd_prompt}, {d} view"], [neg_prompt], 1))
    text_z = torch.stack(text_z, dim=0)

    kd_min = torch.tensor(args.kd_min, dtype=torch.float32, device="cuda")
    kd_max = torch.tensor(args.kd_max, dtype=torch.float32, device="cuda")
    ks_min = torch.tensor(args.ks_min, dtype=torch.float32, device="cuda")
    ks_max = torch.tensor(args.ks_max, dtype=torch.float32, device="cuda")
    nrm_min = torch.tensor(args.nrm_min, dtype=torch.float32, device="cuda")
    nrm_max = torch.tensor(args.nrm_max, dtype=torch.float32, device="cuda")
    nrm_t = get_template_normal(h=args.uv_res, w=args.uv_res)

    glctx = get_glctx()

    for step in tqdm(range(args.n_iter + 1)):
        cur_iter_frac = step / args.n_iter
        losses = {}
        optim.zero_grad()
        lgt.build_mips()
        with torch.no_grad():
            mesh = copy.deepcopy(mesh_t)

        net_output = net(network_input)
        pred_tex = net_output.permute(0, 2, 3, 1)
        pred_kd = pred_tex[..., :-6]
        pred_ks = pred_tex[..., -6:-3]
        pred_n = F.normalize((pred_tex[..., -3:] * 2.0 - 1.0) + nrm_t, dim=-1)

        pred_material = Material(
            {
                "bsdf": "pbr",
                "kd": Texture2D(pred_kd, min_max=[kd_min, kd_max]),
                "ks": Texture2D(pred_ks, min_max=[ks_min, ks_max]),
                "normal": Texture2D(pred_n, min_max=[nrm_min, nrm_max]),
            }
        )
        pred_material["kd"].clamp_()
        pred_material["ks"].clamp_()
        pred_material["normal"].clamp_()
        mesh.material = pred_material

        cam = sample_view_obj(args.n_view, cam_radius=3.25)
        buffers = render_mesh(
            glctx,
            mesh,
            cam["mvp"],
            cam["campos"],
            lgt,
            cam["resolution"],
            spp=cam["spp"],
            msaa=True,
            background=None,
            bsdf="pbr",
        )
        pred_obj_rgb = buffers["shaded"][..., 0:3].permute(0, 3, 1, 2).contiguous()
        pred_obj_ws = buffers["shaded"][..., 3].unsqueeze(1)
        obj_image = pred_obj_rgb * pred_obj_ws + (1 - pred_obj_ws) * args.bg

        all_pos, all_neg = [], []
        text_z_iter = text_z[cam["direction"]]
        for emb in text_z_iter:
            pos, neg = emb.chunk(2)
            all_pos.append(pos)
            all_neg.append(neg)
        text_embedding = torch.cat(all_pos + all_neg, dim=0)

        sd_min_step = compute_sd_step(args.sd_min_l, args.sd_min_r, cur_iter_frac)
        sd_max_step = compute_sd_step(args.sd_max_l, args.sd_max_r, cur_iter_frac)
        sd_loss = guidance.batch_train_step(
            text_embedding,
            obj_image,
            guidance_scale=args.gd_scale,
            min_step=sd_min_step,
            max_step=sd_max_step,
        )

        losses["L_sds"] = sd_loss.item()
        sd_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.sd_max_grad_norm)
        optim.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if step % args.log_freq == 0 and args.logging:
            with torch.no_grad():
                report_process(step, losses, sample_name)
                save_mtl(os.path.join(sample_output_dir, "mesh.mtl"), mesh.material, step=step)
                torchvision.utils.save_image(obj_image[0], os.path.join(sample_output_dir, f"obj_{step:04}.jpg"))

    with torch.no_grad():
        vis_mesh = copy.deepcopy(mesh_t)
        final_pred = net(network_input)
        final_tex = final_pred.permute(0, 2, 3, 1).contiguous()

        final_kd = final_tex[..., :-6]
        final_ks = final_tex[..., -6:-3]
        final_n = F.normalize((final_tex[..., -3:] * 2.0 - 1.0) + nrm_t, dim=-1)

        final_material = Material(
            {
                "bsdf": "pbr",
                "kd": Texture2D(final_kd, min_max=[kd_min, kd_max]),
                "ks": Texture2D(final_ks, min_max=[ks_min, ks_max]),
                "normal": Texture2D(final_n, min_max=[nrm_min, nrm_max]),
            }
        )
        final_material["kd"].clamp_()
        final_material["ks"].clamp_()
        final_material["normal"].clamp_()
        vis_mesh.material = final_material

        export_compact_assets(sample_output_dir, vis_mesh, final_material, keep_only_required=args.keep_only_required)
        if args.render_final_views:
            render_final_views(args, vis_mesh, lgt, sample_output_dir)
        # Final cleanup after all optional renders to enforce compact outputs.
        if args.keep_only_required:
            cleanup_sample_output(sample_output_dir)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        return obj.item()
    except Exception:
        return str(obj)


def path_if_exists(path):
    return os.path.abspath(path) if path and os.path.exists(path) else ""


def build_result_row(obj_id, sample_dir, caption_short=None, caption_long=None, caption_used=None):
    sample_dir = os.path.abspath(sample_dir)
    row = {
        "obj_id": obj_id,
        "mesh": path_if_exists(os.path.join(sample_dir, "mesh.obj")),
        "albedo": path_if_exists(os.path.join(sample_dir, "albedo.png")),
        "rough": path_if_exists(os.path.join(sample_dir, "roughness.png")),
        "metal": path_if_exists(os.path.join(sample_dir, "metallic.png")),
        "normal": path_if_exists(os.path.join(sample_dir, "normal.png")),
    }
    if caption_short is not None:
        row["caption_short"] = caption_short
    if caption_long is not None:
        row["caption_long"] = caption_long
    if caption_used is not None:
        row["caption_used"] = caption_used
    return row


def load_batch_from_tsv(tsv_path, caption_field):
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")
    with open(tsv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if "mesh" not in fieldnames and "mesh_path" not in fieldnames:
            raise ValueError(
                "TSV must contain 'mesh' or 'mesh_path' column. "
                f"Found: {', '.join(fieldnames) if fieldnames else '(none)'}"
            )
        if caption_field not in fieldnames:
            raise ValueError(
                f"TSV missing caption_field '{caption_field}'. "
                f"Found: {', '.join(fieldnames) if fieldnames else '(none)'}"
            )
        rows = [row for row in reader]
    return rows


def check_sample_completed(textures_dir, obj_id):
    sample_dir = os.path.join(textures_dir, obj_id)
    required_files = ["roughness.png", "metallic.png", "albedo.png", "mesh.obj"]
    return all(os.path.isfile(os.path.join(sample_dir, f)) for f in required_files)


def append_to_manifest(tsv_path, new_rows):
    if not new_rows:
        return

    existing_ids = set()
    existing_rows = []
    if os.path.isfile(tsv_path):
        with open(tsv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                existing_ids.add(row.get("obj_id", ""))
                existing_rows.append(row)

    rows_to_add = [r for r in new_rows if r.get("obj_id", "") not in existing_ids]
    if not rows_to_add:
        print(f"[INFO] No new rows to append to manifest {tsv_path}")
        return

    all_rows = existing_rows + rows_to_add
    fieldnames = ["obj_id", "mesh", "albedo", "rough", "metal", "normal"]
    for name in ["caption_short", "caption_long", "caption_used"]:
        if any(name in r for r in all_rows):
            fieldnames.append(name)

    os.makedirs(os.path.dirname(tsv_path) or ".", exist_ok=True)
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"[INFO] Appended {len(rows_to_add)} new rows to manifest {tsv_path} (total: {len(all_rows)})")


def save_experiment_config(exp_dir, args, processed_samples, skipped_samples=None, manifest_path=None, timing_info=None):
    cfg = {
        "options": to_jsonable(vars(args)),
        "ckpt_path": None,
        "tsv_path": os.path.abspath(args.tsv_path) if args.tsv_path else None,
        "save_image": args.logging,
        "processed_samples": processed_samples,
    }
    if skipped_samples:
        cfg["skipped_samples"] = skipped_samples
    if manifest_path:
        cfg["result_tsv"] = os.path.abspath(manifest_path)
    if timing_info:
        cfg["timing"] = timing_info

    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def merge_and_save_experiment_config(exp_dir, args, new_processed, new_skipped=None, manifest_path=None, timing_info=None):
    config_path = os.path.join(exp_dir, "config.json")
    existing_config = None
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            existing_config = json.load(f)

    if existing_config is None:
        save_experiment_config(exp_dir, args, new_processed, new_skipped, manifest_path, timing_info)
        return

    prev_processed = existing_config.get("processed_samples", [])
    existing_ids = {s["obj_id"] for s in prev_processed}
    for sample in new_processed:
        if sample["obj_id"] not in existing_ids:
            prev_processed.append(sample)
            existing_ids.add(sample["obj_id"])
    prev_processed.sort(key=lambda x: x["obj_id"])

    new_processed_ids = {s["obj_id"] for s in new_processed}
    prev_skipped = existing_config.get("skipped_samples", [])
    remaining_skipped = [s for s in prev_skipped if s["obj_id"] not in new_processed_ids]
    existing_skipped_ids = {s["obj_id"] for s in remaining_skipped}
    for s in (new_skipped or []):
        if s["obj_id"] not in existing_skipped_ids:
            remaining_skipped.append(s)
    remaining_skipped.sort(key=lambda x: x["obj_id"])

    timing_key = "timing"
    i = 2
    while timing_key in existing_config:
        timing_key = f"timing{i}"
        i += 1

    existing_config["options"] = to_jsonable(vars(args))
    existing_config["ckpt_path"] = None
    existing_config["tsv_path"] = os.path.abspath(args.tsv_path) if args.tsv_path else None
    existing_config["save_image"] = args.logging
    existing_config["processed_samples"] = prev_processed
    existing_config["skipped_samples"] = remaining_skipped
    if manifest_path:
        existing_config["result_tsv"] = os.path.abspath(manifest_path)
    if timing_info:
        existing_config[timing_key] = timing_info

    os.makedirs(exp_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(existing_config, f, indent=2, ensure_ascii=False)
    print(
        f"[INFO] Updated config.json: {len(prev_processed)} processed, "
        f"{len(remaining_skipped)} skipped, timing key: {timing_key}"
    )


def parse_gpu_ids(gpu_ids_str):
    gpu_ids_str = gpu_ids_str.strip()
    if gpu_ids_str.startswith("[") and gpu_ids_str.endswith("]"):
        gpu_ids_str = gpu_ids_str[1:-1]
    return [int(x.strip()) for x in gpu_ids_str.split(",") if x.strip()]


def estimate_workers_per_gpu(gpu_id, model_memory_gb=8.0, safety_margin=0.85):
    if not torch.cuda.is_available():
        return 1, 0.0
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
    total_memory_gb = total_memory / (1024**3)
    available_gb = total_memory_gb * safety_margin
    estimated = int(available_gb / model_memory_gb)
    workers = max(1, min(estimated, 4))
    return workers, total_memory_gb


def calculate_workers_per_gpu(gpu_ids, workers_per_gpu_str):
    workers_per_gpu_str = str(workers_per_gpu_str).strip().lower()
    if workers_per_gpu_str == "auto":
        if not gpu_ids:
            return 1
        workers, total_mem = estimate_workers_per_gpu(gpu_ids[0])
        print(f"[INFO] Auto-detected GPU memory: {total_mem:.1f} GB")
        print(f"[INFO] Auto-calculated workers_per_gpu: {workers}")
        return workers
    try:
        return max(1, int(workers_per_gpu_str))
    except ValueError:
        print(f"[WARN] Invalid workers_per_gpu '{workers_per_gpu_str}', using default 1")
        return 1


def create_guidance(args):
    guidance = StableDiffusion(device, min=args.sd_min, max=args.sd_max)
    guidance.eval()
    for p in guidance.parameters():
        p.requires_grad = False
    return guidance


def resolve_caption(row, caption_field):
    caption = (row.get(caption_field) or "").strip()
    if caption:
        return caption
    for fallback in ["caption_long", "caption_short", "prompt", "text_prompt"]:
        caption = (row.get(fallback) or "").strip()
        if caption:
            return caption
    return ""


def resolve_obj_id(row, mesh_path, fallback_idx):
    obj_id = (row.get("obj_id") or "").strip()
    if obj_id:
        return obj_id
    if mesh_path:
        return os.path.splitext(os.path.basename(mesh_path))[0]
    return f"sample_{fallback_idx}"


def run_single_gpu_worker(gpu_id, worker_id, rows_subset, args, tsv_dir, textures_dir):
    worker_tag = f"[GPU {gpu_id} W{worker_id}]"
    if torch.cuda.is_available():
        print(f"{worker_tag} CUDA visible count: {torch.cuda.device_count()}, current device: {torch.cuda.current_device()}")
    guidance = create_guidance(args)

    processed_samples = []
    skipped_samples = []

    for local_idx, (global_idx, row) in enumerate(rows_subset):
        mesh_path = (row.get("mesh") or row.get("mesh_path") or "").strip()
        caption_short = (row.get("caption_short") or "").strip()
        caption_long = (row.get("caption_long") or "").strip()
        caption = resolve_caption(row, args.caption_field)
        obj_id = resolve_obj_id(row, mesh_path, global_idx)

        if not mesh_path or not caption:
            skipped_samples.append({"obj_id": obj_id, "reason": "missing mesh or caption"})
            print(f"{worker_tag} Skip row {global_idx}: missing mesh or caption (obj_id={obj_id})")
            continue

        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(tsv_dir, mesh_path)

        sample_output_dir = os.path.join(textures_dir, obj_id)
        os.makedirs(sample_output_dir, exist_ok=True)

        print(f"{worker_tag} Processing {obj_id} ({local_idx + 1}/{len(rows_subset)}, global {global_idx + 1})")
        try:
            run_single_asset(args, guidance, mesh_path, caption, sample_output_dir, obj_id)
            processed_samples.append(
                build_result_row(
                    obj_id,
                    sample_output_dir,
                    caption_short=caption_short,
                    caption_long=caption_long,
                    caption_used=args.caption_field,
                )
            )
        except Exception as e:
            print(f"{worker_tag} Error processing {obj_id}: {e}")
            import traceback

            traceback.print_exc()
            skipped_samples.append({"obj_id": obj_id, "reason": str(e)})

    return processed_samples, skipped_samples


def worker_subprocess_entry():
    gpu_id = int(os.environ["PAINTIT_GPU_ID"])
    worker_id = int(os.environ.get("PAINTIT_WORKER_ID", "0"))
    config_file = os.environ["PAINTIT_CONFIG_FILE"]

    worker_tag = f"[GPU {gpu_id} W{worker_id}]"
    print(f"{worker_tag} Starting subprocess...")
    print(f"{worker_tag} CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    with open(config_file, "rb") as f:
        config = pickle.load(f)

    args_dict = config["args_dict"]
    rows_subset = config["rows_subset"]
    tsv_dir = config["tsv_dir"]
    textures_dir = config["textures_dir"]
    result_file = config["result_file"]

    args = argparse.Namespace(**args_dict)
    processed_samples, skipped_samples = run_single_gpu_worker(gpu_id, worker_id, rows_subset, args, tsv_dir, textures_dir)

    results = {
        "gpu_id": gpu_id,
        "worker_id": worker_id,
        "processed": processed_samples,
        "skipped": skipped_samples,
    }
    with open(result_file, "wb") as f:
        pickle.dump(results, f)
    print(f"{worker_tag} Finished. Processed {len(processed_samples)}, skipped {len(skipped_samples)}")


def run_multi_gpu(args, batch_rows, tsv_dir, textures_dir, gpu_ids, workers_per_gpu):
    num_gpus = len(gpu_ids)
    num_samples = len(batch_rows)
    total_workers = num_gpus * workers_per_gpu

    worker_assignments = [[] for _ in range(total_workers)]
    for idx, row in enumerate(batch_rows):
        worker_idx = idx % total_workers
        worker_assignments[worker_idx].append((idx, row))

    print(
        f"[INFO] Distributing {num_samples} samples across {num_gpus} GPUs x "
        f"{workers_per_gpu} workers = {total_workers} total workers"
    )
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        gpu_total = sum(len(worker_assignments[gpu_idx * workers_per_gpu + w]) for w in range(workers_per_gpu))
        print(f"  GPU {gpu_id}: {gpu_total} samples ({workers_per_gpu} workers)")

    args_dict = vars(args).copy()
    temp_dir = tempfile.mkdtemp(prefix="paintit_multiGPU_")
    print(f"[INFO] Using temp directory: {temp_dir}")

    processes = []
    result_files = []
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        for local_worker_id in range(workers_per_gpu):
            global_worker_id = gpu_idx * workers_per_gpu + local_worker_id
            rows_subset = worker_assignments[global_worker_id]
            if not rows_subset:
                continue

            config_file = os.path.join(temp_dir, f"config_gpu{gpu_id}_worker{local_worker_id}.pkl")
            result_file = os.path.join(temp_dir, f"result_gpu{gpu_id}_worker{local_worker_id}.pkl")
            result_files.append((gpu_id, local_worker_id, result_file))

            config = {
                "args_dict": args_dict,
                "rows_subset": rows_subset,
                "tsv_dir": tsv_dir,
                "textures_dir": textures_dir,
                "result_file": result_file,
            }
            with open(config_file, "wb") as f:
                pickle.dump(config, f)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PAINTIT_GPU_ID"] = str(gpu_id)
            env["PAINTIT_WORKER_ID"] = str(local_worker_id)
            env["PAINTIT_CONFIG_FILE"] = config_file

            cmd = [sys.executable, "-c", "from paint_it import worker_subprocess_entry; worker_subprocess_entry()"]
            print(f"[INFO] Launching subprocess for GPU {gpu_id} Worker {local_worker_id} ({len(rows_subset)} samples)...")
            p = subprocess.Popen(cmd, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
            processes.append((gpu_id, local_worker_id, p))

    print(f"[INFO] Waiting for {len(processes)} subprocesses to complete...")
    for gpu_id, local_worker_id, p in processes:
        return_code = p.wait()
        if return_code != 0:
            print(f"[WARN] Subprocess for GPU {gpu_id} Worker {local_worker_id} exited with code {return_code}")

    all_processed = []
    all_skipped = []
    for gpu_id, local_worker_id, result_file in result_files:
        if os.path.exists(result_file):
            with open(result_file, "rb") as f:
                results = pickle.load(f)
            all_processed.extend(results["processed"])
            all_skipped.extend(results["skipped"])
            print(
                f"[INFO] Collected results from GPU {gpu_id} Worker {local_worker_id}: "
                f"{len(results['processed'])} processed, {len(results['skipped'])} skipped"
            )
        else:
            print(f"[WARN] Result file not found for GPU {gpu_id} Worker {local_worker_id}: {result_file}")

    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"[WARN] Failed to clean up temp directory: {e}")

    all_processed.sort(key=lambda x: x["obj_id"])
    all_skipped.sort(key=lambda x: x["obj_id"])
    return all_processed, all_skipped


def run_batch(args):
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.abspath("./logs/paintit_batch")
    os.makedirs(output_dir, exist_ok=True)
    textures_dir = os.path.join(output_dir, "textures")
    os.makedirs(textures_dir, exist_ok=True)

    if args.result_tsv:
        result_tsv_path = args.result_tsv
        if not os.path.isabs(result_tsv_path):
            result_tsv_path = os.path.abspath(os.path.join(output_dir, result_tsv_path))
    else:
        result_tsv_path = os.path.join(output_dir, "generated_manifest.tsv")
    result_tsv_path = os.path.abspath(result_tsv_path)
    args.result_tsv = result_tsv_path

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    num_gpus = min(args.num_gpus, len(gpu_ids)) if args.num_gpus > 0 else len(gpu_ids)
    gpu_ids = gpu_ids[:num_gpus]
    if not gpu_ids:
        raise RuntimeError("No GPU IDs provided. Set --gpu_ids and --num_gpus.")

    workers_per_gpu = calculate_workers_per_gpu(gpu_ids, args.workers_per_gpu)
    print(f"[INFO] Using {len(gpu_ids)} GPU(s): {gpu_ids}")
    print(f"[INFO] Workers per GPU: {workers_per_gpu}")

    inference_start_time = time.time()
    inference_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] Inference started at: {inference_start_datetime}")

    tsv_dir = os.path.dirname(os.path.abspath(args.tsv_path))
    batch_rows = load_batch_from_tsv(args.tsv_path, args.caption_field)
    total_rows = len(batch_rows)

    if args.max_samples > 0 and args.max_samples < total_rows:
        batch_rows = batch_rows[: args.max_samples]
        print(f"[INFO] Loaded {total_rows} rows from {args.tsv_path}, processing first {args.max_samples} samples")
    else:
        print(f"[INFO] Loaded {total_rows} rows from {args.tsv_path}")
    print(f"[INFO] Using caption field: {args.caption_field}")

    if args.auto_skip:
        auto_skipped_ids = []
        rows_to_process = []
        for row in batch_rows:
            obj_id = resolve_obj_id(row, (row.get("mesh") or row.get("mesh_path") or "").strip(), -1)
            if obj_id and check_sample_completed(textures_dir, obj_id):
                auto_skipped_ids.append(obj_id)
            else:
                rows_to_process.append(row)
        if auto_skipped_ids:
            print(f"[INFO] Auto-skipped {len(auto_skipped_ids)} already-completed samples")
            print(f"[INFO] Remaining samples to process: {len(rows_to_process)}")
        batch_rows = rows_to_process

    total_workers = len(gpu_ids) * workers_per_gpu
    if len(batch_rows) == 0:
        print("[INFO] All samples are already completed. Nothing to do.")
        processed_samples = []
        skipped_samples = []
    elif total_workers > 1 and len(batch_rows) > 1:
        print(
            f"[INFO] Running in parallel mode: {len(gpu_ids)} GPUs x {workers_per_gpu} workers = "
            f"{total_workers} total workers"
        )
        processed_samples, skipped_samples = run_multi_gpu(args, batch_rows, tsv_dir, textures_dir, gpu_ids, workers_per_gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        print(f"[INFO] Running in single-worker mode on GPU {gpu_ids[0]}")
        rows_subset = list(enumerate(batch_rows))
        processed_samples, skipped_samples = run_single_gpu_worker(gpu_ids[0], 0, rows_subset, args, tsv_dir, textures_dir)

    if processed_samples:
        append_to_manifest(result_tsv_path, processed_samples)

    inference_end_time = time.time()
    inference_end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time_seconds = inference_end_time - inference_start_time
    total_time_str = str(timedelta(seconds=int(total_time_seconds)))
    num_samples = len(processed_samples)
    avg_time_per_sample = total_time_seconds / num_samples if num_samples > 0 else 0

    timing_info = {
        "start_time": inference_start_datetime,
        "end_time": inference_end_datetime,
        "total_seconds": round(total_time_seconds, 2),
        "total_time_formatted": total_time_str,
        "num_samples_processed": num_samples,
        "avg_seconds_per_sample": round(avg_time_per_sample, 2),
        "num_gpus": len(gpu_ids),
        "workers_per_gpu": workers_per_gpu,
        "total_workers": len(gpu_ids) * workers_per_gpu,
    }

    print("\n" + "=" * 60)
    print("[TIMING] Inference completed!")
    print(f"[TIMING] Start time: {inference_start_datetime}")
    print(f"[TIMING] End time: {inference_end_datetime}")
    print(f"[TIMING] Total time: {total_time_str} ({total_time_seconds:.2f} seconds)")
    print(f"[TIMING] Samples processed: {num_samples}")
    print(f"[TIMING] Average time per sample: {avg_time_per_sample:.2f} seconds")
    print(f"[TIMING] GPUs used: {len(gpu_ids)}, Workers per GPU: {workers_per_gpu}")
    print("=" * 60 + "\n")

    manifest_for_config = result_tsv_path if os.path.isfile(result_tsv_path) else None
    merge_and_save_experiment_config(
        output_dir,
        args,
        processed_samples,
        skipped_samples,
        manifest_path=manifest_for_config,
        timing_info=timing_info,
    )


def run_single(args):
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    gpu_id = gpu_ids[0] if gpu_ids else 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[INFO] Running single sample mode on GPU {gpu_id}")

    if args.output_dir:
        sample_output_dir = os.path.abspath(args.output_dir)
    else:
        exp_name = "_".join(args.prompt.split(" "))
        sample_output_dir = os.path.join("./logs", f"{time.strftime('%Y%m%d', time.localtime())}_{exp_name}")

    guidance = create_guidance(args)
    run_single_asset(args, guidance, args.mesh_path, args.prompt, sample_output_dir, args.texture_name)
    print(f"[INFO] Saved outputs to {os.path.abspath(sample_output_dir)}")


if __name__ == "__main__":
    args = parse_args()
    if args.tsv_path:
        run_batch(args)
    else:
        run_single(args)
