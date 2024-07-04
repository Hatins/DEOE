# Detecting Every Object from Events
## Framework
![Framework](./git_figs/framework.png)

## Comparasion with the RVT in close-set setting (training with pedestrians and cars)
![Open class: bicycle](https://github.com/Hatins/DEOE/blob/main/gifs/bicycle.gif)

![Open class: motorcycle and truck](https://github.com/Hatins/DEOE/blob/main/gifs/truck_motorcar.gif)

## Installation
We recommend using cuda11.8 to avoid unnecessary environment problem.
```
conda create -y -n deoe python=3.11
conda activate deoe

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1

pip install pytorch_lightning==1.8.6 wandb pandas plotly opencv-python tabulate pycocotools
bbox-visualizer StrEnum hydra-core einops torchdata tqdm numba h5py hdf5plugin lovely-tensors
tensorboardX pykeops scikit-learn ipdb timm
```

## Required Data
### DSEC
You could download the processed DSEC-Detection by click here.

### GEN4
You could get the raw GEN4 in [RVT](https://github.com/uzh-rpg/RVT).
And get the processed data by following the [Instruction](https://github.com/uzh-rpg/RVT/blob/master/scripts/genx/README.md) proposed by RVT.
Note that to keep the labels for all the classes following [here](https://github.com/uzh-rpg/RVT/issues/4).
