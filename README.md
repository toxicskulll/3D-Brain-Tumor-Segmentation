# 3D Brain Tumor Segmentation using nnU-Net

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle-blue.svg)](https://kaggle.com)

A state-of-the-art implementation of 3D brain tumor segmentation using NVIDIA's optimized U-Net architecture on the BraTS 2021 dataset. This project demonstrates advanced medical image analysis techniques with comprehensive explanations for educational purposes.

## 🧠 Project Overview

Brain tumor segmentation is a critical task in medical imaging that helps radiologists and oncologists identify and measure tumor regions for treatment planning. This project implements an end-to-end pipeline for automated brain tumor segmentation using deep learning.

### Key Features

- **NVIDIA-Optimized Architecture**: Implements NVIDIA's exact U-Net specifications with deep supervision
- **BraTS 2021 Dataset**: Uses the latest Brain Tumor Segmentation Challenge dataset
- **Multi-Modal MRI Processing**: Handles FLAIR, T1, T1CE, and T2 MRI modalities
- **Advanced Training Techniques**: Mixed precision training, deep supervision, and cosine annealing
- **Comprehensive Documentation**: Detailed explanations for educational purposes
- **Production-Ready Code**: Clean, well-documented, and optimized implementation

## 🎯 Segmentation Targets

The model segments three critical tumor regions:

1. **Whole Tumor (WT)**: Complete tumor area including all sub-regions
2. **Tumor Core (TC)**: Tumor without the peritumoral edema
3. **Enhancing Tumor (ET)**: Actively enhancing tumor region

## 🏗️ Architecture

### NVIDIA U-Net Specifications
- **Input**: 5 channels (4 MRI modalities + 1 one-hot encoded)
- **Output**: 3 channels (WT, TC, ET)
- **Depth**: 7-level encoder-decoder with skip connections
- **Filters**: [64, 96, 128, 192, 256, 384, 512]
- **Normalization**: Instance normalization
- **Activation**: LeakyReLU (slope=0.01)
- **Parameters**: ~31M trainable parameters

### Key Technical Features
- **Deep Supervision**: Multiple auxiliary losses for better gradient flow
- **Mixed Precision Training**: Faster training with AMP
- **Advanced Loss Function**: Combined Dice + Cross-Entropy with class weighting
- **Data Augmentation**: Geometric and intensity transformations

## 📊 Dataset Information

### BraTS 2021 Dataset
- **Patients**: 1,251 training cases
- **Modalities**: FLAIR, T1, T1CE, T2
- **Resolution**: 240 × 240 × 155 voxels
- **Voxel Spacing**: 1mm³ isotropic
- **Preprocessing**: Skull-stripped and co-registered

### Label Mapping
- **0**: Background (healthy brain tissue)
- **1**: Necrotic and non-enhancing tumor core
- **2**: Peritumoral edema
- **3**: GD-enhancing tumor (remapped from label 4)

## 🚀 Getting Started

### Prerequisites

```bash
# Core dependencies
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
matplotlib>=3.4.0
nibabel>=3.2.0
scikit-learn>=1.0.0

# Medical imaging
monai>=0.8.0
scipy>=1.7.0

# Utilities
tqdm>=4.62.0
joblib>=1.1.0
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/toxicskulll/3D-Brain-Tumor-Segmentation.git
cd 3D-Brain-Tumor-Segmentation
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download BraTS 2021 dataset**:
   - Register at [BraTS Challenge](http://braintumorsegmentation.org/)
   - Download the training data
   - Place in `/kaggle/input/brats-2021-task1/` (for Kaggle) or update paths

### Quick Start

1. **Open the enhanced notebook**:
```bash
jupyter notebook enhanced-brain-tumor-seg-3d-nnunet.ipynb
```

2. **Follow the step-by-step guide**:
   - Data extraction and preprocessing
   - Model architecture setup
   - Training configuration
   - Model training and evaluation

## 📚 Notebook Structure

### 1. Environment Setup and Configuration
- Library imports and system configuration
- GPU setup and memory management
- Path configuration for Kaggle environment

### 2. Data Extraction and Processing
- **BraTS dataset extraction** from compressed archives
- **Train/validation split** (80/20) with reproducible random seed
- **Memory optimization** strategies for large medical datasets
- **Data structure verification** and sample exploration

### 3. nnU-Net Data Preprocessing
- **Multi-modal combination** into single 4D volumes
- **Label remapping** from [0,1,2,4] to [0,1,2,3]
- **File organization** into nnU-Net expected format
- **Quality assurance** and preprocessing verification

### 4. Deep Learning Model Architecture
- **NVIDIA U-Net implementation** with exact specifications
- **BraTS-specific loss function** combining Dice and Cross-Entropy
- **Deep supervision setup** for improved training
- **Architecture testing** and parameter counting

### 5. Training Setup and Execution
- **NVIDIA training configuration** with optimal hyperparameters
- **Mixed precision training** setup with AMP
- **Learning rate scheduling** with cosine annealing
- **Model checkpointing** and progress monitoring

## 🔧 Configuration Options

### Model Configuration
```python
model_config = {
    'in_channels': 5,           # 4 modalities + 1 OHE
    'out_channels': 3,          # WT, TC, ET
    'filters': [64, 96, 128, 192, 256, 384, 512],
    'normalization': 'instance',
    'deep_supervision': True
}
```

### Training Configuration
```python
training_config = {
    'learning_rate': 0.0003,    # NVIDIA optimal LR
    'epochs': 30,               # Full training epochs
    'batch_size': 1,            # Limited by GPU memory
    'scheduler': True,          # Cosine annealing
    'amp': True,                # Mixed precision
    'gradient_clipping': True,  # Stability
    'save_checkpoints': True
}
```

## 📈 Expected Results

### Performance Metrics
- **Dice Score**: 0.85-0.92 (depending on region)
- **Hausdorff Distance**: <10mm for most cases
- **Training Time**: ~8-12 hours on V100 GPU
- **Inference Time**: ~30 seconds per case

### Validation Performance
- **Whole Tumor**: Dice ~0.90
- **Tumor Core**: Dice ~0.85
- **Enhancing Tumor**: Dice ~0.80

## 🛠️ Advanced Usage

### Custom Dataset
To use your own dataset:

1. **Organize data** in BraTS format:
```
data/
├── Patient001/
│   ├── Patient001_flair.nii.gz
│   ├── Patient001_t1.nii.gz
│   ├── Patient001_t1ce.nii.gz
│   ├── Patient001_t2.nii.gz
│   └── Patient001_seg.nii.gz
└── Patient002/
    └── ...
```

2. **Update paths** in the notebook
3. **Adjust preprocessing** if needed

### Hyperparameter Tuning
Key parameters to experiment with:
- **Learning rate**: 0.0001 - 0.001
- **Loss weights**: Dice vs Cross-Entropy balance
- **Augmentation strength**: Rotation, scaling, intensity
- **Architecture depth**: Filter progression

### Multi-GPU Training
For distributed training:
```python
# Enable DataParallel
model = nn.DataParallel(model)

# Or use DistributedDataParallel
model = nn.parallel.DistributedDataParallel(model)
```

## 📊 Monitoring and Visualization

### Training Metrics
- **Loss curves**: Training and validation loss
- **Dice scores**: Per-region segmentation quality
- **Learning rate**: Schedule visualization
- **GPU utilization**: Memory and compute usage

### Segmentation Visualization
- **Multi-modal display**: All 4 MRI modalities
- **Overlay visualization**: Predictions on original images
- **3D rendering**: Volume visualization of tumors
- **Comparison plots**: Ground truth vs predictions

## 🔬 Technical Details

### Memory Management
- **Gradient checkpointing**: Reduce memory usage
- **Mixed precision**: FP16 for forward, FP32 for backward
- **Batch size optimization**: Balance speed vs memory
- **Data loading**: Efficient I/O with prefetching

### Optimization Techniques
- **Deep supervision**: Auxiliary losses at multiple scales
- **Label smoothing**: Reduce overconfidence
- **Gradient clipping**: Prevent exploding gradients
- **Weight decay**: L2 regularization

### Data Augmentation
- **Geometric**: Rotation, flipping, scaling
- **Intensity**: Normalization, contrast adjustment
- **Spatial**: Elastic deformations
- **Noise injection**: Gaussian noise for robustness

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Slow Training**:
   - Check GPU utilization
   - Optimize data loading
   - Use multiple workers

3. **Poor Convergence**:
   - Adjust learning rate
   - Check data preprocessing
   - Verify loss function

4. **NaN Loss**:
   - Enable gradient clipping
   - Check for invalid data
   - Reduce learning rate

### Performance Optimization
- **Use SSD storage** for faster data loading
- **Optimize batch size** for your GPU memory
- **Enable mixed precision** for 2x speedup
- **Use multiple GPUs** for distributed training

## 📖 Educational Resources

### Medical Imaging Concepts
- **MRI Modalities**: Understanding FLAIR, T1, T1CE, T2
- **Brain Anatomy**: Tumor types and locations
- **Segmentation Metrics**: Dice, IoU, Hausdorff distance
- **Medical Image Formats**: NIfTI, DICOM standards

### Deep Learning Concepts
- **U-Net Architecture**: Encoder-decoder with skip connections
- **Loss Functions**: Dice loss for segmentation
- **Optimization**: Adam, learning rate scheduling
- **Regularization**: Dropout, weight decay, normalization

### Implementation Details
- **PyTorch Fundamentals**: Tensors, autograd, modules
- **MONAI Framework**: Medical imaging toolkit
- **Mixed Precision**: Automatic mixed precision training
- **Model Checkpointing**: Saving and loading models

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution
- **New architectures**: Transformer-based segmentation
- **Data augmentation**: Advanced augmentation techniques
- **Evaluation metrics**: Additional performance measures
- **Visualization**: Better result visualization tools
- **Documentation**: Improved explanations and tutorials

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BraTS Challenge**: For providing the dataset and evaluation framework
- **NVIDIA**: For the optimized U-Net architecture specifications
- **MONAI Team**: For the excellent medical imaging toolkit
- **PyTorch Team**: For the deep learning framework
- **Medical Imaging Community**: For advancing the field

## 📞 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [@toxicskulll](https://github.com/toxicskulll)
- **LinkedIn**: [Your LinkedIn Profile]

## 📚 References

1. Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature Methods (2021).

2. Menze, B. H., et al. "The multimodal brain tumor image segmentation benchmark (BRATS)." IEEE Transactions on Medical Imaging (2015).

3. Bakas, S., et al. "Advancing the cancer genome atlas glioma MRI collections with expert segmentation labels and radiomic features." Scientific Data (2017).

4. Ronneberger, O., Fischer, P., & Brox, T. "U-net: Convolutional networks for biomedical image segmentation." MICCAI (2015).

5. Myronenko, A. "3D MRI brain tumor segmentation using autoencoder regularization." BrainLes Workshop (2018).

---

⭐ **Star this repository** if you find it helpful!

🐛 **Report issues** on our [GitHub Issues](https://github.com/toxicskulll/3D-Brain-Tumor-Segmentation/issues) page.

📖 **Read our documentation** for detailed implementation guides.