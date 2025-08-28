# Domain-Adversarial Learning for Robust Image Classification

Unsupervised domain adaptation for improving deep learning model generalization on distorted images using adversarial training techniques.

## Overview

This project tackles the critical challenge of model robustness in computer vision by implementing an unsupervised domain adaptation method that enables ResNet50 to maintain high accuracy on distorted images without requiring labeled data from the target domain. By leveraging adversarial training with a gradient reversal layer, the model learns domain-invariant features that generalize across both clean and distorted image distributions, achieving **67% accuracy on distorted images** compared to only 25% with vanilla ResNet50.

## Architecture

<p align="center">
  <img src="docs/images/architecture.png" alt="Domain Adaptation Architecture" width="800">
</p>

*Domain-Adversarial Neural Network architecture featuring dual classifiers with gradient reversal layer for learning domain-invariant features*

### Key Components:
- **Feature Extractor (Gf)**: ResNet50 backbone for deep convolutional features
- **Label Predictor (Gy)**: Classifies images into 1000 ImageNet categories
- **Domain Classifier (Gd)**: Discriminates between source (clean) and target (distorted) domains
- **Gradient Reversal Layer**: Flips gradients during backpropagation to maximize domain confusion

## Distortion Types

<p align="center">
  <img src="docs/images/distortions.png" alt="Image Distortion Examples" width="900">
</p>

*Ten distortion techniques applied to test images: grayscale, false-color, high/low-pass filtering, noise, rotations, contrast adjustments, and salt-and-pepper noise*

## Performance Results

### Quantitative Results

| Model | Undistorted Images | Distorted Images | Improvement |
|-------|-------------------|------------------|-------------|
| Vanilla ResNet50 | 72% | 25% | - |
| Domain Adaptation Model | 70% | **67%** | **+42%** |

### Training Curves

<p align="center">
  <img src="docs/images/training_curves.png" alt="Training Curves" width="800">
</p>

*Loss and accuracy curves showing model convergence on both source and target domains*

## Key Features

- **Unsupervised Adaptation**: No labels required for distorted images during training
- **Gradient Reversal**: Novel backpropagation technique for domain confusion
- **Multi-Distortion Robustness**: Handles 10+ types of image distortions simultaneously
- **ImageNet Scale**: Trained on ImageNet-mini with 1000 classes
- **Plug-and-Play**: Compatible with any CNN backbone architecture
- **Real-World Applications**: Extends to medical imaging, satellite imagery, and low-quality cameras

## Technical Implementation

### Domain Adaptation Training Process

1. **Source Domain Training**: Minimize classification loss on clean images
2. **Domain Discrimination**: Train discriminator to distinguish clean vs distorted features
3. **Adversarial Learning**: Update feature extractor to fool the discriminator
4. **Feature Alignment**: Achieve domain-invariant representation through minimax optimization

### Loss Functions

```python
# Total loss with adversarial component
L_total = L_classification(source) - λ * L_domain(source, target)

# Classification loss (CrossEntropy)
L_classification = CrossEntropyLoss(predictions, labels)

# Domain confusion loss (Binary CrossEntropy)
L_domain = BCELoss(domain_predictions, domain_labels)
```

## Installation

### Prerequisites
```bash
# Python 3.6+ required
python --version

# Install PyTorch
pip install torch>=1.4.0 torchvision>=0.5.0

# Install additional dependencies
pip install -r requirements.txt
```

### Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/domain-adversarial-learning.git
cd domain-adversarial-learning
```

## Dataset Preparation

### ImageNet-mini Structure
```
imagenet-mini/
├── train/                    # Clean training images
│   ├── n01440764/           # Class folder (tench fish)
│   ├── n01443537/           # Class folder (goldfish)
│   └── ...                  # 1000 classes total
├── val/                     # Clean validation images
├── test_undistorted/        # Clean test images
└── test_distorted/          # Distorted test images (10 types)
```

### Download Dataset
```bash
# Download from Google Drive (publicly available)
gdown --folder https://drive.google.com/drive/folders/1a8yqwGKm5Jo7cPLHw69rjfCaQRH5Rfff
```

### Generate Distorted Images
```bash
# Apply 10 distortion types to test images
python data_manipulation.py \
  --input_dir imagenet-mini/test_undistorted \
  --output_dir imagenet-mini/test_distorted
```

## Usage

### Training Vanilla ResNet50 (Baseline)
```bash
python resnet50_imagenet.py \
  --train_path "imagenet-mini/train" \
  --val_path "imagenet-mini/val" \
  --epochs 20 \
  --batch_size 48
```

### Training Domain Adaptation Model
```bash
python domain_adaptation.py \
  --train_path "imagenet-mini/train" \
  --val_path "imagenet-mini/test_distorted" \
  --epochs 15 \
  --alpha 0.2 \
  --batch_size 48
```

### Testing on Undistorted Images
```bash
python test_undistorted.py \
  --test_path "imagenet-mini/test_undistorted" \
  --checkpoint_dir "checkpoints/res50model_checkpoint.pth"
```

### Testing on Distorted Images
```bash
# Vanilla ResNet50
python test_distorted.py \
  --test_path "imagenet-mini/test_distorted" \
  --checkpoint_dir "checkpoints/res50model_checkpoint.pth"

# Domain Adaptation Model
python test_distorted.py \
  --test_path "imagenet-mini/test_distorted" \
  --checkpoint_dir "checkpoints/dom_ada_checkpoint.pth"
```

## Model Architecture Details

### Feature Extractor
- **Base Model**: ResNet50 (pretrained on ImageNet)
- **Output**: 2048-dimensional feature vector
- **Modifications**: Removed final classification layer

### Domain Discriminator
```python
discriminator = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
```

### Gradient Reversal Layer
```python
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.001 | Adam optimizer |
| Batch Size | 48 | Limited by GPU memory |
| Epochs (Baseline) | 20 | Vanilla ResNet50 training |
| Epochs (DA) | 15 | Domain adaptation training |
| α (Alpha) | 0.2 | Domain confusion weight |
| Momentum | 0.9 | SGD optimizer momentum |

## Distortion Implementation

### Supported Distortions
1. **Grayscale**: RGB to grayscale conversion
2. **False Color**: Opponent color transformation
3. **Low Contrast**: Contrast reduction to 10%
4. **Uniform Noise**: Additive noise (width=0.1)
5. **Salt & Pepper**: Random pixel corruption (p=0.1)
6. **High-Pass Filter**: Gaussian HPF (σ=3)
7. **Low-Pass Filter**: Gaussian LPF (σ=10)
8. **Rotation 90°/180°/270°**: Geometric transformations

## Results Analysis

### Domain Confusion Success
The domain discriminator's inability to distinguish between source and target domains (loss approaching 0.5) indicates successful feature alignment:

<p align="center">
  <img src="docs/images/domain_loss.png" alt="Domain Loss Convergence" width="600">
</p>

### t-SNE Visualization
Feature distributions become more aligned after domain adaptation training, showing improved generalization capability.

## Applications

This domain adaptation technique is particularly valuable for:

- **Medical Imaging**: Adapting models across different scanners/protocols
- **Autonomous Vehicles**: Handling weather/lighting variations
- **Satellite Imagery**: Cross-sensor generalization
- **Quality Control**: Robust inspection under varying conditions
- **Mobile Photography**: Compensating for low-quality cameras

## Future Improvements

- [ ] Implement multi-source domain adaptation
- [ ] Add attention mechanisms for feature alignment
- [ ] Explore cycle-consistent adversarial adaptation
- [ ] Extend to semantic segmentation tasks
- [ ] Develop online adaptation capabilities
- [ ] Create browser-based demo with TensorFlow.js


## References

1. [Ganin et al., 2016] Domain-Adversarial Training of Neural Networks, JMLR
2. [Geirhos et al., 2018] Generalisation in Humans and Deep Neural Networks, NeurIPS
3. [Tzeng et al., 2017] Adversarial Discriminative Domain Adaptation, CVPR

## Team

**Group 27** - Johns Hopkins University

- Vikram Shivakumar (vshivak1@jhu.edu)
- Marcelo Morales (lmoral10@jhu.edu)  
- Bhupendra Mahar (bmahar1@jhu.edu)
- **Ritwik Rohan** (rrohan2@jhu.edu)

**Mentors**: Yuta Kobayashi (CA), Jingfeng Wu (TA)

## Acknowledgments

- Johns Hopkins CS482/682 Machine Learning course staff
- PyTorch team for the deep learning framework
- ImageNet team for the dataset
- Authors of the original DANN paper for the methodology

---

*Improving model robustness one domain at a time* 🎯
