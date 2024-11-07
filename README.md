# CCCMDT
Compositional Conditional Consistency Model Diffusion Transformer
## Installation
- Python > 3.9 recommended Conda
- PyTorch >= 1.13.0+cu11.7
```
conda create -n cccmdt python=3.9
conda activate cccmdt
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/MT121201/CCCMDT.git
cd CCCMDT
pip install -r requirements.txt
```