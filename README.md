# FGEFNet 

## Installation
* Clone this repo into a directory 
* Organize your datasets as required
* Install Python dependencies. We use python 3.8.10 and pytorch 1.8.1
```
pip install -r requirements.txt
```


## Organize the counting dataset
We use a list file to collect all the images and their ground truth annotations in a counting dataset. When your dataset is organized as recommended in the following, the format of this list file is defined as:
```
train/scene01/img01.jpg train/scene01/img01.txt
train/scene01/img02.jpg train/scene01/img02.txt
...
train/scene02/img01.jpg train/scene02/img01.txt
```

### Dataset structures:
```
DATA_ROOT/
        |->train/
        |    |->scene01/
        |    |->scene02/
        |    |->...
        |->test/
        |    |->scene01/
        |    |->scene02/
        |    |->...
        |->train.list
        |->test.list
```
DATA_ROOT is your path containing the counting datasets.

### Annotations format
For the annotations of each image, we use a single txt file which contains one annotation per line. Note that indexing for pixel values starts at 0. The expected format of each line is:
```
x1 y1
x2 y2
...
```

## Training

The network can be trained using the `train.py` script. For training on SHTechPartA, use

```
CUDA_VISIBLE_DEVICES=0 python train.py  --eval --data_root $DATA_ROOT \
    --dataset_file SHHA \
    --epochs 3500 \
    --lr_drop 3500 \
    --output_dir ./logs \
    --checkpoints_dir ./weights \
    --tensorboard_dir ./logs \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --eval_freq 1 \
    --gpu_id 0
```
By default, a periodic evaluation will be conducted on the validation set.

## Testing

A trained model (with an MAE of **49.1**) on SHTechPartA is available at "./ckpt", run the following commands to launch a visualization demo:

```
CUDA_VISIBLE_DEVICES=0 python run_test.py --weight_path ./weights/03-31_18-23_SHHA_vgg16_bn_0.0001_best_mae --output_dir ./logs/
```

