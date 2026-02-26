#!/bin/bash
# =============================================================================
# Full Reproduction Script for cknna_action_proprio_simplerenv.csv
# =============================================================================
#
# Starting from a fresh machine with:
#   - NVIDIA GPU (H100 80GB recommended)
#   - CUDA 12.x drivers installed
#   - conda installed at $CONDA_ROOT (default: ~/conda)
#   - git, git-lfs, gsutil available
#   - HuggingFace CLI logged in (huggingface-cli login)
#
# This script clones all repos, downloads all checkpoints, creates all conda
# environments, and runs the full 3-phase CKNNA pipeline for 18 models.
#
# The OpenVLA-7B finetuned model (openvla-7b-bridge-ft-200k) requires a
# separate ~34h training run -- see Section 7.
#
# Total time estimate (excluding training): ~4-6 hours on 1xH100
#
# Usage:
#   bash reproduce_cknna_pipeline.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-${WORK}/conda}"
export WORK CONDA_ROOT
CONDA="${CONDA_ROOT}/bin/conda"

mkdir -p "${WORK}"
cd "${WORK}"


# =========================================================================
# SECTION 1: Clone all repositories
# =========================================================================
echo "========================================"
echo " Section 1: Cloning repositories"
echo "========================================"

# 1a. StarVLA
if [ ! -d "${WORK}/starVLA/.git" ]; then
    git clone https://github.com/yunfeixie233/starVLA.git "${WORK}/starVLA"
    cd "${WORK}/starVLA" && git checkout starVLA && cd "${WORK}"
fi

# 1b. SimplerEnv-OpenVLA
if [ ! -d "${WORK}/SimplerEnv-OpenVLA/.git" ]; then
    git clone https://github.com/yunfeixie233/SimplerEnv-OpenVLA.git "${WORK}/SimplerEnv-OpenVLA"
    cd "${WORK}/SimplerEnv-OpenVLA"
    git submodule update --init --recursive
    cd "${WORK}"
fi

# 1c. Octo (submodule of SimplerEnv, pinned commit)
if [ ! -d "${WORK}/SimplerEnv-OpenVLA/octo/.git" ]; then
    cd "${WORK}/SimplerEnv-OpenVLA"
    git clone https://github.com/octo-models/octo.git octo
    cd octo && git checkout 653c54ac && cd "${WORK}"
fi

# 1d. lerobot (provides compute_cknna.py)
if [ ! -d "${WORK}/lerobot/.git" ]; then
    git clone https://github.com/yunfeixie233/lerobot.git "${WORK}/lerobot"
fi

# 1e. openvla (for LoRA finetuning, Section 7)
if [ ! -d "${WORK}/openvla/.git" ]; then
    git clone https://github.com/openvla/openvla.git "${WORK}/openvla"
fi

# 1f. Isaac-GR00T (N1.5) -- fork with Bridge embodiment tag
if [ ! -d "${WORK}/Isaac-GR00T/.git" ]; then
    git clone https://github.com/yunfeixie233/Isaac-GR00T.git "${WORK}/Isaac-GR00T"
    cd "${WORK}/Isaac-GR00T" && git checkout cknna-bridge && cd "${WORK}"
fi

# 1g. Isaac-GR00T (N1.6 -- separate directory)
mkdir -p "${WORK}/gr00t_1p6"
if [ ! -d "${WORK}/gr00t_1p6/Isaac-GR00T/.git" ]; then
    git clone https://github.com/NVIDIA/Isaac-GR00T.git "${WORK}/gr00t_1p6/Isaac-GR00T"
fi

# 1h. CogACT
if [ ! -d "${WORK}/CogACT/.git" ]; then
    git clone https://github.com/microsoft/CogACT.git "${WORK}/CogACT"
fi


# =========================================================================
# SECTION 2: Download checkpoints and weights
# =========================================================================
echo ""
echo "========================================"
echo " Section 2: Downloading checkpoints"
echo "========================================"

# 2a. StarVLA finetuned checkpoints (HuggingFace -> playground/Pretrained_models/)
PRETRAINED="${WORK}/starVLA/playground/Pretrained_models"
mkdir -p "${PRETRAINED}"

declare -A STARVLA_HF_REPOS
STARVLA_HF_REPOS=(
    ["Qwen-FAST-Bridge-RT-1"]="StarVLA/Qwen-FAST-Bridge-RT-1"
    ["Qwen-OFT-Bridge-RT-1"]="StarVLA/Qwen-OFT-Bridge-RT-1"
    ["Qwen-GR00T-Bridge"]="StarVLA/Qwen-GR00T-Bridge"
    ["Qwen-GR00T-Bridge-RT-1"]="StarVLA/Qwen-GR00T-Bridge-RT-1"
    ["Qwen3VL-GR00T-Bridge-RT-1"]="StarVLA/Qwen3VL-GR00T-Bridge-RT-1"
    ["Qwen3VL-OFT-Bridge-RT-1"]="StarVLA/Qwen3VL-OFT-Bridge-RT-1"
)

for name in "${!STARVLA_HF_REPOS[@]}"; do
    dest="${PRETRAINED}/${name}"
    if [ ! -d "${dest}" ]; then
        echo "Downloading ${STARVLA_HF_REPOS[$name]} -> ${dest}"
        huggingface-cli download "${STARVLA_HF_REPOS[$name]}" --local-dir "${dest}"
    else
        echo "[${name}] already exists, skipping."
    fi
done

# 2b. StarVLA base VLMs
for vlm in "Qwen/Qwen2.5-VL-3B-Instruct" "Qwen/Qwen3-VL-4B-Instruct"; do
    short=$(basename "${vlm}")
    dest="${PRETRAINED}/${short}"
    if [ ! -d "${dest}" ]; then
        echo "Downloading ${vlm} -> ${dest}"
        huggingface-cli download "${vlm}" --local-dir "${dest}"
    else
        echo "[${short}] already exists, skipping."
    fi
done

# 2c. StarVLA FAST tokenizer
if [ ! -d "${PRETRAINED}/fast" ]; then
    echo "Downloading physical-intelligence/fast tokenizer"
    huggingface-cli download physical-intelligence/fast --local-dir "${PRETRAINED}/fast"
fi

# 2d. StarVLA Action-tuned base VLM (used by FAST and some GR00T variants)
if [ ! -d "${PRETRAINED}/Qwen2.5-VL-3B-Instruct-Action" ]; then
    echo "Downloading StarVLA/Qwen2.5-VL-3B-Instruct-Action"
    huggingface-cli download StarVLA/Qwen2.5-VL-3B-Instruct-Action \
        --local-dir "${PRETRAINED}/Qwen2.5-VL-3B-Instruct-Action"
fi

# 2e. Update StarVLA checkpoint configs to point to local base_vlm paths
echo "Patching StarVLA checkpoint config.yaml base_vlm paths..."
for name in "${!STARVLA_HF_REPOS[@]}"; do
    cfg="${PRETRAINED}/${name}/config.yaml"
    if [ -f "${cfg}" ]; then
        old_vlm=$(grep -oP 'base_vlm:\s*\K\S+' "${cfg}" 2>/dev/null || true)
        if [ -n "${old_vlm}" ]; then
            vlm_basename=$(basename "${old_vlm}")
            sed -i "s|base_vlm:.*|base_vlm: ${PRETRAINED}/${vlm_basename}|" "${cfg}"
            echo "  ${name}: base_vlm -> ${PRETRAINED}/${vlm_basename}"
        fi
    fi
done

# 2f. OpenVLA-7B base model
OPENVLA_CKPT="${WORK}/SimplerEnv-OpenVLA/checkpoints/openvla-7b"
if [ ! -d "${OPENVLA_CKPT}" ]; then
    echo "Downloading openvla/openvla-7b"
    huggingface-cli download openvla/openvla-7b --local-dir "${OPENVLA_CKPT}"
fi

# 2g. RT-1-X checkpoint (from Google Cloud Storage)
RT1X_CKPT="${WORK}/SimplerEnv-OpenVLA/checkpoints/rt_1_x_tf_trained_for_002272480_step"
if [ ! -d "${RT1X_CKPT}" ]; then
    echo "Downloading RT-1-X checkpoint from GCS"
    cd "${WORK}/SimplerEnv-OpenVLA/checkpoints"
    gsutil -m cp -r \
        gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip .
    unzip rt_1_x_tf_trained_for_002272480_step.zip
    rm -f rt_1_x_tf_trained_for_002272480_step.zip
    cd "${WORK}"
fi

# 2h. GR00T N1.5 bridge-finetuned checkpoint
if [ ! -d "${WORK}/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2" ]; then
    echo "Downloading GR00T N1.5 Bridge checkpoint"
    git lfs install
    git clone https://huggingface.co/ShuaiYang03/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2 \
        "${WORK}/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2"
fi

# 2i. GR00T N1.6 bridge-finetuned checkpoint
if [ ! -d "${WORK}/GR00T-N1.6-bridge" ]; then
    echo "Downloading GR00T N1.6 Bridge checkpoint"
    huggingface-cli download nvidia/GR00T-N1.6-bridge \
        --local-dir "${WORK}/GR00T-N1.6-bridge"
fi

# 2j. SpatialVLA, Pi0, Octo, CogACT -- auto-downloaded at runtime
echo ""
echo "The following models auto-download from HF during feature extraction:"
echo "  - IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge (SpatialVLA)"
echo "  - HaomingSong/lerobot-pi0-bridge (Pi0 lerobot)"
echo "  - rail-berkeley/octo-base-1.5 (Octo-base)"
echo "  - CogACT/CogACT-Small, CogACT/CogACT-Base, CogACT/CogACT-Large"
echo ""


# =========================================================================
# SECTION 3: Create conda environments
# =========================================================================
echo ""
echo "========================================"
echo " Section 3: Creating conda environments"
echo "========================================"

# ---------- 3a. starVLA ----------
if ! ${CONDA} env list | grep -q "starVLA"; then
    echo "Creating starVLA env..."
    ${CONDA} create -n starVLA python=3.10 -y
    source "${CONDA_ROOT}/bin/activate" starVLA
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install flash-attn --no-build-isolation
    cd "${WORK}/starVLA"
    pip install -r requirements.txt
    cd "${WORK}"
else
    echo "starVLA env already exists."
fi

# ---------- 3b. simpler_env ----------
if ! ${CONDA} env list | grep -q "simpler_env"; then
    echo "Creating simpler_env..."
    ${CONDA} create -n simpler_env python=3.10 -y
    source "${CONDA_ROOT}/bin/activate" simpler_env
    pip install numpy==1.24.4
    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
    pip install "setuptools<75"
    cd "${WORK}/SimplerEnv-OpenVLA/ManiSkill2_real2sim"
    pip install -e .
    cd "${WORK}/SimplerEnv-OpenVLA"
    pip install -e .
    pip install tensorflow==2.15.0
    pip install timm==0.9.10 tokenizers==0.15.2 accelerate==0.32.1
    pip install transformers==4.40.1
    pip install flash-attn==2.6.1 --no-build-isolation
    pip install gymnasium==0.29.1
    pip install sapien==2.2.2
    # RT-1-X deps
    pip install tensorflow_hub tensorflow_datasets rlds tf_agents
    # Octo deps
    pip install --upgrade "jax[cuda12_pip]==0.4.20" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install nvidia-cudnn-cu12==8.9.2.26 nvidia-nccl-cu12==2.20.5 \
        nvidia-cuda-cupti-cu12==12.4.127 nvidia-cusolver-cu12==11.6.3.83 \
        nvidia-cuda-runtime-cu12==12.4.127 nvidia-cublas-cu12==12.1.3.1
    cd "${WORK}/SimplerEnv-OpenVLA/octo"
    pip install -e .
    cd "${WORK}"
else
    echo "simpler_env env already exists."
fi

# ---------- 3c. spatialvla_env ----------
if ! ${CONDA} env list | grep -q "spatialvla_env"; then
    echo "Creating spatialvla_env..."
    ${CONDA} create -n spatialvla_env python=3.10 -y
    source "${CONDA_ROOT}/bin/activate" spatialvla_env
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install "transformers>=4.47.0" accelerate pillow huggingface_hub
else
    echo "spatialvla_env env already exists."
fi

# ---------- 3d. openvla_env ----------
if ! ${CONDA} env list | grep -q "openvla_env"; then
    echo "Creating openvla_env..."
    ${CONDA} create -n openvla_env python=3.10 -y
    source "${CONDA_ROOT}/bin/activate" openvla_env
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.40.1 timm==0.9.16 accelerate pillow huggingface_hub
    pip install flash-attn==2.6.1 --no-build-isolation
    pip install peft==0.11.1 draccus==0.8.0 dlimp rich
    pip install tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3
    pip install einops wandb sentencepiece==0.1.99 jsonlines json-numpy
    cd "${WORK}/openvla"
    pip install -e . --no-deps
    cd "${WORK}"
else
    echo "openvla_env env already exists."
fi

# ---------- 3e. groot_libero (for GR00T N1.5 + lerobot models) ----------
if ! ${CONDA} env list | grep -q "groot_libero"; then
    echo "Creating groot_libero env..."
    ${CONDA} create -n groot_libero python=3.10 -y
    source "${CONDA_ROOT}/bin/activate" groot_libero
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    cd "${WORK}/lerobot"
    pip install -e ".[groot]"
    pip install h5py
    cd "${WORK}"
else
    echo "groot_libero env already exists."
fi

# ---------- 3f. pi0fast_env (for Pi0 lerobot models) ----------
if ! ${CONDA} env list | grep -q "pi0fast_env"; then
    echo "Creating pi0fast_env..."
    ${CONDA} create -n pi0fast_env python=3.10 -y
    source "${CONDA_ROOT}/bin/activate" pi0fast_env
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    cd "${WORK}/lerobot"
    pip install -e ".[groot]"
    pip install h5py
    pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
    cd "${WORK}"
else
    echo "pi0fast_env env already exists."
fi

# ---------- 3g. groot16 (GR00T N1.6 via uv) ----------
if ! ${CONDA} env list | grep -q "groot16"; then
    echo "Creating groot16 env..."
    ${CONDA} create -n groot16 python=3.10 -y
    source "${CONDA_ROOT}/bin/activate" groot16
    pip install uv
    cd "${WORK}/gr00t_1p6/Isaac-GR00T"
    uv sync --python 3.10
    cd "${WORK}"
else
    echo "groot16 env already exists."
fi

# ---------- 3h. cogact ----------
if ! ${CONDA} env list | grep -q "cogact"; then
    echo "Creating cogact env..."
    ${CONDA} create -n cogact python=3.10 -y
    source "${CONDA_ROOT}/bin/activate" cogact
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install transformers timm accelerate pillow huggingface_hub
    cd "${WORK}/CogACT"
    pip install -e .
    cd "${WORK}"
else
    echo "cogact env already exists."
fi


# =========================================================================
# SECTION 4: Set up data symlink
# =========================================================================
echo ""
echo "========================================"
echo " Section 4: Setting up data symlink"
echo "========================================"

CKNNA_DATA="${WORK}/starVLA/cknna/cknna_data"
SIMPLER_CKNNA="${WORK}/SimplerEnv-OpenVLA/cknna/cknna_data"

mkdir -p "${CKNNA_DATA}"

if [ ! -L "${SIMPLER_CKNNA}" ]; then
    ln -sf "${CKNNA_DATA}" "${SIMPLER_CKNNA}"
    echo "Created symlink: ${SIMPLER_CKNNA} -> ${CKNNA_DATA}"
else
    echo "Symlink already exists."
fi


# =========================================================================
# SECTION 5: Run CKNNA Pipeline
# =========================================================================
echo ""
echo "========================================"
echo " Section 5: Running CKNNA pipeline"
echo "========================================"

# ---------- Phase 1: Download Bridge data (run once) ----------
echo "--- Phase 1: Bridge data (5000 samples) ---"
source "${CONDA_ROOT}/bin/activate" starVLA
cd "${WORK}/starVLA"

if [ -f "${CKNNA_DATA}/feats_B.pt" ] && [ -f "${CKNNA_DATA}/metadata.json" ]; then
    echo "Phase 1 data already exists, skipping."
else
    python cknna/load_bridge_data.py \
        --output_dir "${CKNNA_DATA}" \
        --num_samples 5000 \
        --num_chunks 3 \
        --seed 42
fi

# ---------- Phase 2A: StarVLA feature extraction ----------
echo ""
echo "--- Phase 2A: StarVLA models (6 checkpoints) ---"
source "${CONDA_ROOT}/bin/activate" starVLA
cd "${WORK}/starVLA"
bash cknna/run_cknna_starvla.sh

# ---------- Phase 2A-action: StarVLA action representations ----------
echo ""
echo "--- Phase 2A-action: StarVLA action repr ---"
bash cknna/run_action_repr_starvla.sh

# ---------- Phase 2B: SimplerEnv models ----------
echo ""
echo "--- Phase 2B: SimplerEnv models (SpatialVLA, Pi0, OpenVLA) ---"
bash "${WORK}/SimplerEnv-OpenVLA/cknna/run_cknna_simplerenv.sh"

# ---------- Phase 2B-action: SimplerEnv action representations ----------
echo ""
echo "--- Phase 2B-action: SimplerEnv action repr ---"
bash "${WORK}/SimplerEnv-OpenVLA/cknna/run_action_repr_simplerenv.sh"

# ---------- Phase 2C: RT-1-X ----------
echo ""
echo "--- Phase 2C: RT-1-X ---"
source "${CONDA_ROOT}/bin/activate" simpler_env
LD_LIBRARY_PATH="${CONDA_ROOT}/envs/simpler_env/lib:${LD_LIBRARY_PATH:-}" \
python "${WORK}/SimplerEnv-OpenVLA/cknna/extract_features_rt1x.py" \
    --ckpt "${WORK}/SimplerEnv-OpenVLA/checkpoints/rt_1_x_tf_trained_for_002272480_step" \
    --data_dir "${CKNNA_DATA}" \
    --output_dir "${CKNNA_DATA}/rt1x-bridge"

# ---------- Phase 2D: Octo-base ----------
echo ""
echo "--- Phase 2D: Octo-base ---"
python "${WORK}/SimplerEnv-OpenVLA/cknna/extract_features_octo.py" \
    --ckpt "hf://rail-berkeley/octo-base-1.5" \
    --data_dir "${CKNNA_DATA}" \
    --output_dir "${CKNNA_DATA}/octo-base-bridge"

# ---------- Phase 2E: CogACT (Small, Base, Large) ----------
echo ""
echo "--- Phase 2E: CogACT ---"
source "${CONDA_ROOT}/bin/activate" cogact
for SIZE in Small Base Large; do
    LOWER=$(echo "${SIZE}" | tr '[:upper:]' '[:lower:]')
    OUT="${CKNNA_DATA}/cogact-${LOWER}-bridge"
    if [ -f "${OUT}/feats_A.pt" ]; then
        echo "[cogact-${SIZE}] exists, skipping."
        continue
    fi
    python "${WORK}/SimplerEnv-OpenVLA/cknna/extract_features_cogact.py" \
        --ckpt "CogACT/CogACT-${SIZE}" \
        --data_dir "${CKNNA_DATA}" \
        --output_dir "${OUT}"
done

# ---------- Phase 2F: GR00T N1.5 ----------
echo ""
echo "--- Phase 2F: GR00T N1.5 ---"
source "${CONDA_ROOT}/bin/activate" groot_libero
PYTHONPATH="${WORK}/Isaac-GR00T" \
python "${WORK}/SimplerEnv-OpenVLA/cknna/extract_features_groot_n15.py" \
    --ckpt "${WORK}/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2" \
    --data_dir "${CKNNA_DATA}" \
    --output_dir "${CKNNA_DATA}/groot-n15-bridge"

PYTHONPATH="${WORK}/Isaac-GR00T" \
python "${WORK}/SimplerEnv-OpenVLA/cknna/extract_action_repr_groot_n15.py" \
    --ckpt "${WORK}/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2" \
    --data_dir "${CKNNA_DATA}" \
    --output_dir "${CKNNA_DATA}/groot-n15-bridge"

# ---------- Phase 2G: GR00T N1.6 ----------
echo ""
echo "--- Phase 2G: GR00T N1.6 ---"
source "${CONDA_ROOT}/bin/activate" groot16
PYTHONPATH="${WORK}/gr00t_1p6/Isaac-GR00T" \
python "${WORK}/SimplerEnv-OpenVLA/cknna/extract_features_groot_n16.py" \
    --ckpt "${WORK}/GR00T-N1.6-bridge" \
    --data_dir "${CKNNA_DATA}" \
    --output_dir "${CKNNA_DATA}/groot-n16-bridge"

PYTHONPATH="${WORK}/gr00t_1p6/Isaac-GR00T" \
python "${WORK}/SimplerEnv-OpenVLA/cknna/extract_action_repr_groot_n16.py" \
    --ckpt "${WORK}/GR00T-N1.6-bridge" \
    --data_dir "${CKNNA_DATA}" \
    --output_dir "${CKNNA_DATA}/groot-n16-bridge"


# =========================================================================
# SECTION 6: Compute final CKNNA scores
# =========================================================================
echo ""
echo "========================================"
echo " Section 6: Final CKNNA computation"
echo "========================================"

source "${CONDA_ROOT}/bin/activate" starVLA

# Proprio CKNNA: all feats_A vs feats_B
FEATS_A_PATHS=""
for dir in "${CKNNA_DATA}"/*/; do
    if [ -f "${dir}feats_A.pt" ]; then
        FEATS_A_PATHS="${FEATS_A_PATHS} ${dir}feats_A.pt"
    fi
done

python "${WORK}/SimplerEnv-OpenVLA/cknna/compute_cknna.py" \
    --feats_A ${FEATS_A_PATHS} \
    --feats_B "${CKNNA_DATA}/feats_B.pt" \
    --topk 5 10 20 \
    --also_mutual_knn \
    --output "${CKNNA_DATA}/cknna_results_proprio_all.json"

# Action CKNNA: per-model feats_A vs feats_action
for dir in "${CKNNA_DATA}"/*/; do
    fa="${dir}feats_A.pt"
    fact="${dir}feats_action.pt"
    if [ -f "${fa}" ] && [ -f "${fact}" ]; then
        name=$(basename "${dir}")
        echo "CKNNA action: ${name}"
        python "${WORK}/SimplerEnv-OpenVLA/cknna/compute_cknna.py" \
            --feats_A "${fa}" \
            --feats_B "${fact}" \
            --topk 5 10 20 \
            --also_mutual_knn \
            --output "${dir}cknna_action_repr.json"
    fi
done


echo ""
echo "========================================"
echo " CKNNA Pipeline Complete"
echo "========================================"
echo ""
echo "Proprio results: ${CKNNA_DATA}/cknna_results_proprio_all.json"
echo "Action results:  ${CKNNA_DATA}/<model>/cknna_action_repr.json"
echo ""
echo "To compile the final CSV, merge these JSON files with WidowX SimplerEnv"
echo "success rates into: starVLA/cknna/record/cknna_action_proprio_simplerenv.csv"
echo ""
echo "NOTE: The openvla-7b-bridge-ft-200k row requires a separate training step."
echo "See Section 7 below."
echo ""


# =========================================================================
# SECTION 7: OpenVLA LoRA Fine-tuning (optional, ~34h on 1xH100)
# =========================================================================
# Uncomment the block below to run OpenVLA LoRA finetuning on Bridge V2.
# After training, extract features and compute CKNNA for the finetuned model.
#
# source "${CONDA_ROOT}/bin/activate" openvla_env
# cd "${WORK}/openvla"
#
# # Download Bridge V2 dataset (RLDS/TFDS format)
# python -c "
# import tensorflow_datasets as tfds
# builder = tfds.builder_from_directory('bridge_orig')
# builder.download_and_prepare()
# "
#
# # Run LoRA finetuning (~34h on 1xH100)
# PYTHONNOUSERSITE=1 \
# LD_LIBRARY_PATH=${CONDA_ROOT}/envs/openvla_env/lib:${LD_LIBRARY_PATH:-} \
# TOKENIZERS_PARALLELISM=false \
# torchrun --standalone --nnodes 1 --nproc-per-node 1 \
#   vla-scripts/finetune.py \
#   --vla_path ${WORK}/SimplerEnv-OpenVLA/checkpoints/openvla-7b \
#   --data_root_dir ${WORK}/openvla/datasets \
#   --dataset_name bridge_orig \
#   --run_root_dir ${WORK}/openvla/runs \
#   --adapter_tmp_dir ${WORK}/openvla/adapter-tmp \
#   --lora_rank 32 \
#   --batch_size 16 \
#   --grad_accumulation_steps 1 \
#   --learning_rate 5e-4 \
#   --image_aug true \
#   --wandb_project openvla-bridge-ft \
#   --save_steps 50000 \
#   --max_steps 200000
#
# # The merged checkpoint will be at:
# # ${WORK}/openvla/runs/openvla-7b+bridge_orig+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--200000_chkpt/
# FT_CKPT="${WORK}/openvla/runs/openvla-7b+bridge_orig+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--200000_chkpt"
#
# # Extract features for finetuned model
# source "${CONDA_ROOT}/bin/activate" openvla_env
# python "${WORK}/SimplerEnv-OpenVLA/cknna/extract_features_openvla.py" \
#     --ckpt "${FT_CKPT}" \
#     --data_dir "${CKNNA_DATA}" \
#     --output_dir "${CKNNA_DATA}/openvla-7b-bridge-ft-200k"
#
# python "${WORK}/SimplerEnv-OpenVLA/cknna/extract_action_repr_openvla.py" \
#     --ckpt "${FT_CKPT}" \
#     --data_dir "${CKNNA_DATA}" \
#     --output_dir "${CKNNA_DATA}/openvla-7b-bridge-ft-200k" \
#     --seed 42
#
# # Compute CKNNA for finetuned model
# python "${WORK}/SimplerEnv-OpenVLA/cknna/compute_cknna.py" \
#     --feats_A "${CKNNA_DATA}/openvla-7b-bridge-ft-200k/feats_A.pt" \
#     --feats_B "${CKNNA_DATA}/feats_B.pt" \
#     --topk 5 10 20 --also_mutual_knn \
#     --output "${CKNNA_DATA}/openvla-7b-bridge-ft-200k/cknna_proprio.json"
#
# python "${WORK}/SimplerEnv-OpenVLA/cknna/compute_cknna.py" \
#     --feats_A "${CKNNA_DATA}/openvla-7b-bridge-ft-200k/feats_A.pt" \
#     --feats_B "${CKNNA_DATA}/openvla-7b-bridge-ft-200k/feats_action.pt" \
#     --topk 5 10 20 --also_mutual_knn \
#     --output "${CKNNA_DATA}/openvla-7b-bridge-ft-200k/cknna_action_repr.json"
