# Step 1: 安装Anaconda（Python3.7.3）
```bash
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh;
```

# Step 2: 创建虚拟环境并利用conda和pip安装需要的包
```bash
conda create --name maskrcnn_benchmark
source activate maskrcnn_benchmark

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm requests opencv-contrib-python==3.4.2.17

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
conda install pytorch-nightly=1.0.0 cudatoolkit=9.0 -c pytorch

# install torchvision
pip install torchvision==0.2.0

# install pycocotools
pip install pycocotools==2.0.0

# install PyTorch maskscoring_rcnn
cd ~/github
git clone https://github.com/zjhuang22/maskscoring_rcnn.git
cd maskscoring_rcnn
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop
```
