#!/usr/bin/env bash
# set -euo pipefail

# # 1. 初始化 Conda 环境
# CONDA_BASE="$(conda info --base)"
# source "$CONDA_BASE/etc/profile.d/conda.sh"

# # 激活 paint-it 环境
# conda activate paintit

#（可选）只编译 3090/4090 的算力，缩短时间
export TORCH_CUDA_ARCH_LIST="8.6;8.9"

export CUDA_HOME=/home/pubNAS3/zhiqi/.conda/envs/paintit
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$LIBRARY_PATH
export CPATH="$CONDA_PREFIX/include:${CPATH}"
export CPATH=$CONDA_PREFIX/include/eigen3:$CPATH
export PATH=$PATH:/home/pubNAS3/Github/tools/blender-3.2.2-linux-x64
export PATH=$PATH:/home/pubNAS3/zhiqi/Github/tools/blender-3.2.2-linux-x64
export CPATH=/usr/local/cuda/include:$CONDA_PREFIX/targets/x86_64-linux/include
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib64:$LIBRARY_PATH

#（可选）如果你使用镜像站下载 HF 模型，可保留
export HF_ENDPOINT="https://hf-mirror.com"
export HUGGINGFACE_CO_RESOLVE_ENDPOINT="https://hf-mirror.com"

# ================= 配置区 =================

# 实验名称 (将作为文件夹名创建在 experiments 下)
EXP_NAME="paintit"

# TSV 路径 (建议绝对路径，或相对于 Paint-it 的路径)
BATCH_TSV="../experiments/common_splits/test.tsv"

# 输出根目录 (指向 project_root/experiments/EXP_NAME)
# 假设脚本在 project_root/Paint-it 下运行
OUTPUT_ROOT="../experiments/${EXP_NAME}"

# 文本字段（caption_short 或 caption_long）
CAPTION_FIELD="caption_short"

# 最大处理样本数 (-1 表示处理所有样本)
# 用于快速测试或部分推理
MAX_SAMPLES=2

# 导出与坐标系
MESH_NORM_MODE="texgaussian"   # 与 TexGaussian 坐标系保持一致
KEEP_ONLY_REQUIRED="True"      # 仅保留 albedo/metallic/normal(可选)/roughness/mesh.obj

# 训练/导出细节
N_ITER=1500
LOGGING="True"
RENDER_FINAL_VIEWS="True"
FINAL_RENDER_CHUNK=8

# 结果 manifest 路径（留空则默认 OUTPUT_ROOT/generated_manifest.tsv）
RESULT_TSV=""

# 多GPU配置
# GPU_IDS: 使用的GPU编号，逗号分隔 (例如: "0,1,2,3")
# NUM_GPUS: 实际使用的GPU数量 (会自动取 GPU_IDS 和 NUM_GPUS 的较小值)
# WORKERS_PER_GPU: 每张GPU上并行运行的进程数
#   - "auto": 根据GPU显存自动计算最优值 (推荐)
#   - 数字 (如 "2"): 手动指定固定数量
# GPU_IDS="0,1"
# NUM_GPUS=2
# WORKERS_PER_GPU=2

# 2卡配置 (示例)
GPU_IDS="2,3"
NUM_GPUS=2
WORKERS_PER_GPU=1

# ==========================================

echo "Starting Batch Inference..."
echo "Config: ${BATCH_TSV}"
echo "Output: ${OUTPUT_ROOT}"
echo "Caption: ${CAPTION_FIELD}"
echo "Max Samples: ${MAX_SAMPLES}"
echo "Mesh Norm Mode: ${MESH_NORM_MODE}"
echo "Keep Only Required: ${KEEP_ONLY_REQUIRED}"
echo "GPU IDs: ${GPU_IDS}, Num GPUs: ${NUM_GPUS}, Workers/GPU: ${WORKERS_PER_GPU}"
echo "Total parallel workers: $((NUM_GPUS * WORKERS_PER_GPU))"
echo "Textures will be stored under: ${OUTPUT_ROOT}/textures"

if [[ -n "${RESULT_TSV}" ]]; then
  python3 paint_it.py \
  --tsv_path "${BATCH_TSV}" \
  --caption_field "${CAPTION_FIELD}" \
  --output_dir "${OUTPUT_ROOT}" \
  --result_tsv "${RESULT_TSV}" \
  --max_samples "${MAX_SAMPLES}" \
  --gpu_ids "${GPU_IDS}" \
  --num_gpus "${NUM_GPUS}" \
  --workers_per_gpu "${WORKERS_PER_GPU}" \
  --mesh_norm_mode "${MESH_NORM_MODE}" \
  --keep_only_required "${KEEP_ONLY_REQUIRED}" \
  --n_iter "${N_ITER}" \
  --logging "${LOGGING}" \
  --render_final_views "${RENDER_FINAL_VIEWS}" \
  --final_render_chunk "${FINAL_RENDER_CHUNK}"
else
  python3 paint_it.py \
  --tsv_path "${BATCH_TSV}" \
  --caption_field "${CAPTION_FIELD}" \
  --output_dir "${OUTPUT_ROOT}" \
  --max_samples "${MAX_SAMPLES}" \
  --gpu_ids "${GPU_IDS}" \
  --num_gpus "${NUM_GPUS}" \
  --workers_per_gpu "${WORKERS_PER_GPU}" \
  --mesh_norm_mode "${MESH_NORM_MODE}" \
  --keep_only_required "${KEEP_ONLY_REQUIRED}" \
  --n_iter "${N_ITER}" \
  --logging "${LOGGING}" \
  --render_final_views "${RENDER_FINAL_VIEWS}" \
  --final_render_chunk "${FINAL_RENDER_CHUNK}"
fi
