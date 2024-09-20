
# How to Configure

## 1. Install **conda** and configure the **environment** of conda

Run the following commands to initialize and create a new conda environment:

```bash
conda init
# Ensure the Python version is 3.7.16
conda create -n py37 python=3.7.16 -y 
conda activate py37
```

## 2. Install CUDA Toolkit, PyTorch, and MMCV

Run the following commands to install necessary libraries:

```bash
# Install CUDA Toolkit version 11.3.1
conda install nvidia/label/cuda-11.3.1::cuda-toolkit

# Install PyTorch 1.11.0, torchvision, and torchaudio with CUDA 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# Install MMCV (OpenMMLab) for CUDA 11.3 and PyTorch 1.11.0
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```

## 3. Download **Co-DETR** from GitHub

Clone the repository and install the runtime dependencies:

```bash
# Clone Co-DETR repository
git clone https://github.com/Sense-X/Co-DETR.git

# Change directory to Co-DETR
cd Co-DETR

# Install the required packages
pip install -r requirements/runtime.txt

# missing einops
pip install einops
```

## 4. Move `image_demo.py`

Move the `image_demo.py` file from the configs directory to the root:

```bash
# Move the file
mv CO-DETR/configs/image_demo.py CO-DETR/image_demo.py
```

## 5. Download **Co-DETR weights**

Download the **co-detr-weights** from the following local URL: `http://192.168.1.60:9000` (via [minoadmin]).

## 6. Create the `weights` directory and move the downloaded weights

```bash
# Create the weights directory
mkdir weights

# Move the downloaded weight files into the weights directory
mv ~/Downloads/co-detr-weights/* ./weights
```

## 7. Run the program

Finally, run the program using the following command:

```bash
python image_demo.py demo/demo.jpg projects/configs/co_dino/co_dino_5scale_lsj_swin_large_16e_o365tolvis.py weights/co_dino_5scale_lsj_swin_large_16e_o365tolvis.pth --palette red
```

## 8. Run the Video_demo.py

```bash
python video_demo.py ~/Downloads/can_0710.avi projects/configs/co_dino_vit/co_dino_5scale_lsj_vit_large_lvis.py weights/co_dino_5scale_lsj_vit_large_lvis.pth --out ./output.mp4
```