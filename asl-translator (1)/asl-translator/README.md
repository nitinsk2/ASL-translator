# Real-Time American Sign Language Translator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**CSE 474 – Introduction to Machine Learning, Spring 2026**  
**University at Buffalo**

**Team:** Nitin Suresh Kumar, Yash Sabale, Vanshaj Arora

---

```bash
# Python 3.8+
pip install torch torchvision opencv-python numpy matplotlib seaborn scikit-learn

# Optional: MediaPipe for better hand detection
pip install mediapipe
```


```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/asl-translator.git
cd asl-translator

# Run real-time demo (requires webcam)
cd demo
python realtime_asl_demo.py

# With trained model
python realtime_asl_demo.py --model ../models/asl_mobilenetv2.pth

# With MediaPipe hand detection
python realtime_asl_demo.py --model ../models/asl_mobilenetv2.pth --use-mediapipe
```



```bash
- Model: MobileNetV2 (ImageNet pretrained)
- Optimizer: Adam (lr=1e-3, then 1e-4 after unfreezing)
- Epochs: 10
- Batch size: 32
- Data augmentation: flip, rotation, color jitter
```


```bash
- Model: I3D (Inflated 3D ConvNet)
- Input: 32 frames @ 224x224
- Optimizer: Adam (lr=1e-4)
- Epochs: 30
- Batch size: 8
```