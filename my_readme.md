# How to config

1. Install **conda** and configure **enviroment** of conda.

```shell
conda init
# must be python 3.7.16
conda create -n py37 python=3.7.16 -y 
conda activate py37
```

2. Run the following command in the shell

```shell
conda install nvidia/label/cuda-11.3.1::cuda-toolkit
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```

3. Download [Co-Detr](https://github.com/Sense-X/Co-DETR.git) to local.

```shell
git clone https://github.com/Sense-X/Co-DETR.git
cd Co-DETR
pip install -r requirements/runtime.txt
```

4. Cut `CO-DETR/configs/image_demo.py` to `CO-DETR/image_demo.py`

5. Download **co-detr-weights** in [[minoadmin](192.168.1.60:9000)]

6. Create directory `CO-DETR/weights` and move downloaded file to `weights`

```shell
mkdir weights
mv ~/Downloads/co-detr-weights/* ./weights
```

7. run the program

```shell
python image_demo.py demo/demo.jpg projects/configs/co_dino/co_dino_5scale_lsj_swin_large_16e_o365tolvis.py weights/co_dino_5scale_lsj_swin_large_16e_o365tolvis.pth --palette red 
```