# Detecting Every Object from Events [2025, TPAMI]
## Framework
![Framework](./readme/git_figs/framework.png)

## Comparasion with the RVT in close-set (pedestrians and cars) setting
### Waiting for video loading, or [download](https://github.com/Hatins/DEOE/raw/main/readme/gifs/DEOD.mp4) the mp4 file directly ...
![Open class: bicycle](https://github.com/Hatins/DEOE/blob/main/readme/gifs/videos.gif)

## Installation
We recommend using cuda11.8 to avoid unnecessary environmental problems.
```
conda create -y -n deoe python=3.11

conda activate deoe

pip install torch==2.1.1 torchvision==0.16.1 torchdata==0.7.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

pip install wandb pandas plotly opencv-python tabulate pycocotools bbox-visualizer StrEnum hydra-core einops torchdata tqdm numba h5py hdf5plugin lovely-tensors tensorboardX pykeops scikit-learn ipdb timm opencv-python-headless pytorch_lightning==1.8.6 numpy==1.26.3

pip install openmim

mim install mmcv
```

## Required Data
We recommend using DSEC-Detection training and evaluation first (about 2 days), since 1 Mpx usually takes a long time to train (about 10 days) if you only have a single GPU.
### DSEC
You can download the processed DSEC-Detection by clicking [here](https://drive.google.com/file/d/15k-4tc4m2uhWCUXC5impkkcGlu9Q2t5T/view?usp=drive_link).

### GEN4
You can get the raw GEN4 in [RVT](https://github.com/uzh-rpg/RVT).
And get the processed data by following the [Instruction](https://github.com/uzh-rpg/RVT/blob/master/scripts/genx/README.md) proposed by RVT.
Note that to keep the labels for all the classes following [here](https://github.com/uzh-rpg/RVT/issues/4).

## Checkpoints
<table>
  <tr>
    <th style="text-align:center;"> </th>
    <th style="text-align:center;">DSEC-Detection</th>
    <th style="text-align:center;">GEN4</th>
  </tr>
  <tr>
    <td style="text-align:center;">Pre-trained checkpoints</td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1RdiL1HQQA-Nnt2AmhsuNkDZJIUUHp9kx/view?usp=drive_link">download</a></td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1yjC0cVK23t_wcbyEVepjeYHS1sZXWlny/view?usp=drive_link">download</a></td>
  </tr>
  <tr>
    <td style="text-align:center;">AUC-Unknown</td>
    <td style="text-align:center;">25.1</td>
    <td style="text-align:center;">23.5</td>
  </tr>
</table>

## Evaluation
Before the evaluation, please read some of [important settings](https://github.com/Hatins/DEOE/blob/main/readme/test_setting.md) in our experiments.

Set `DATASET` = `dsec` or `gen4`.

Set `DATADIR` = path to the DSEC-Detection or 1 Mpx dataset directory.

Set `CHECKPOINT` = path to the checkpoint used for evaluation.

```Bash
python validation.py dataset={DATASET} dataset.path={DATADIR} checkpoint={CHECKPOINT} +experiment/{DATASET}='base.yaml'
```
The batchsize, lr, and the other hyperparameters could be adjusted in file `config\experiments\dataset\base.yaml`.

### Evaluation for mixed categories or each category.
Set the `testing_classes` to full categories in file `config\dataset\dataset.yaml`.

Set the `unseen_classes` to the categories evaluated as the unknown categories in file `config\dataset\dataset.yaml`.

The first results outpute by the console are the results for unseen classes, while the second is for testing classes (generally full categories).

### Computed AUC for recall curve.
```Bash
python compute_auc.py
```
## Training
Set `DATASET` = `dsec` or `gen4`.

Set `DATADIR` = path to  the DSEC-Detection or 1 Mpx dataset directory.

```Bash
python train.py dataset={DATASET} dataset.path={DATADIR} +experiment/{DATASET}='base.yaml'
```
The batchsize, lr, and the other hyperparameters could be adjusted in file `config\experiments\dataset\base.yaml`.

## Visualization of results 
Set `DATASET` = `dsec` or `gen4`.

Set `CHECKPOINT` = path to the checkpoint used for evaluation.

Set `h5_file` = path to files used for visualization like `h5_file = /DSEC_process/val/zurich_city_15_a`.

```Bash
python demo.py dataset={DATASET} checkpoint={CHECKPOINT} +experiment/{DATASET}='base.yaml'
```
Then the output images and video will be saved in  folder `DEOE\prediction`.

## Citation
If you find our work is helpful, please considering cite us.
```bibtex
@article{zhang2024detecting,
  title={Detecting Every Object from Events},
  author={Zhang, Haitian and Xu, Chang and Wang, Xinya and Liu, Bingde and Hua, Guang and Yu, Lei and Yang, Wen},
  journal={arXiv preprint arXiv:2404.05285},
  year={2024}
}
```


