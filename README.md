# CT-to-CTA synthesis

Minimal implementation for training and inference with the proposed method.

## Setup

Install a CUDA-compatible PyTorch build and the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Create `data/train.json` following `data/manifest.example.json`. Input volumes
must be registered NIfTI files with matching geometry.

## Run

```bash
# Training
python train_diffusion.py --config config.yaml

# Distributed training
torchrun --standalone --nproc_per_node=2 train_diffusion.py --config config.yaml

# Inference
python inference_diffusion.py \
  --config config.yaml \
  --checkpoint outputs/run/checkpoints/latest.pt \
  --ct_path /path/to/ct.nii.gz \
  --output_dir predictions
```

Data and pretrained weights are not included.
