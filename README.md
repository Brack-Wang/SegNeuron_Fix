# SegNeuron_fix

A deep learning framework for neuron instance segmentation in brain imaging data, based on the SSNS-Net architecture. This project supports pretraining, supervised training, inference, and post-processing for neuron segmentation tasks.

This repo is the fork version of the [SegNeuron](https://github.com/yanchaoz/SegNeuron), fixing bugs while implementing.

## Features

- Self-supervised pretraining with frequency and spatial mixing augmentation
- Supervised training with MNet architecture
- Affinity-based neuron boundary prediction
- Multicut-based instance segmentation post-processing
- Support for CZI and TIFF image formats


## Installation

### Prerequisites

- Linux (tested on Ubuntu)
- NVIDIA GPU with CUDA 12.x support
- Conda package manager

### Setup Environment

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SegNeuron.git
cd SegNeuron
```

2. Create conda environment from the provided environment file:
```bash
conda env create -f environment.yml
conda activate segN
```

Alternatively, you can install key dependencies manually:
```bash
conda create -n segN python=3.9
conda activate segN
conda install pytorch=2.0.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Key Dependencies

- PyTorch 2.0.0+ with CUDA support
- MONAI 1.1.0
- scikit-image 0.20.0
- h5py, tifffile, imageio
- elf (for graph-based segmentation)

## Usage

### 1. Data Preparation

Prepare your training data in TIFF format. The data should include:
- Raw images (grayscale neuron images)
- Ground truth labels (instance segmentation masks)

Use the provided scripts to create synthetic training data if needed:
```bash
# Create synthetic dataset from single neuron traces
python script/synthesis_dataset.py

# Generate segmentation training data
python script/createSgData.py
```

To convert CZI files to TIFF:
```bash
python script/czi2tif.py
```

### 2. Pretraining (Optional)

Pretrain the model with self-supervised learning:
```bash
cd Pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3 python pretrain.py
```

### 3. Supervised Training

Configure the training parameters in `Train_and_Inference/config/SegNeuron.yaml`:
- Set `data_folder` and `data_folder_val` to your data paths
- Adjust `batch_size`, `total_iters`, `base_lr` as needed
- Set `pretrain_path` if using pretrained weights

Run training:
```bash
cd Train_and_Inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python supervised_train.py
```

Training outputs:
- Model checkpoints saved to `./models/`
- TensorBoard logs for monitoring

### 4. Inference

Run affinity prediction on test images:
```bash
cd Train_and_Inference
python inference.py
```

Modify `inference.py` to set:
- Input image path
- Model checkpoint path
- Output directory

### 5. Post-processing

Convert affinity maps to instance segmentation using multicut:
```bash
cd Postprocess
python FRMC_post.py
```

Configure in `FRMC_post.py`:
- `root_path`: directory containing affinity predictions
- `beta`: multicut edge weight parameter (default: 0.25)


## Evaluation Metrics

The framework uses standard segmentation metrics:
- **Adapted Rand Error (ARAND)**: Measures segmentation accuracy
- **Variation of Information (VOI)**: Split and merge errors

## Acknowledgement

This code is based on [SSNS-Net](https://github.com/weih527/SSNS-Net) (IEEE TMI'22) by Huang Wei et al. The post-processing tools are based on [constantinpape/elf](https://github.com/constantinpape/elf).

## License

This project is for research purposes. Please cite the original SSNS-Net paper if you use this code.

## Contact

For questions or issues, please open a GitHub issue.
