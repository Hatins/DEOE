import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path
import shutil
import torch
from torch.backends import cuda, cudnn
from callbacks.custom import get_ckpt_callback, get_viz_callback
cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')
from loggers.utils import get_wandb_logger, get_ckpt_path

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelSummary

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module

import h5py
import hdf5plugin
import ipdb
import numpy as np
import cv2
from tqdm import tqdm

def sort_key(filename):
    return int(filename.split('.')[0])


@hydra.main(config_path='config', config_name='val', version_base='1.2')

def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    gpus = config.hardware.gpus
    assert isinstance(gpus, int), 'no more than 1 GPU supported'
    gpus = 'cuda:0'

    ckpt_path = Path(config.checkpoint)

    module = fetch_model_module(config=config)
    module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config})
    module.set_model_to_gpus(gpus)
    module = module.eval()

    mode = 'pre' #['gt', 'pre']

    h5_file = '/data/zht/DSEC/DSEC_process/val/zurich_city_15_a'

    ev_file = h5_file + '/event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations.h5'

    labels_gt = h5_file + '/labels_v2/labels.npz'



    labels = np.load(labels_gt)['labels']
    frame_to_labels = np.load(labels_gt)['objframe_idx_2_label_idx']
    frame_to_ev = np.load(h5_file + '/event_representations_v2/stacked_histogram_dt=50_nbins=10/objframe_idx_2_repr_idx.npy')

    if mode == 'pre':
        images_out_put_dir = 'predictions/images'
        video_out_put_dir = 'predictions/video'
    elif mode == 'gt':
        images_out_put_dir = 'pre_gt/images'
        video_out_put_dir = 'pre_gt/video'

    if not os.path.exists(images_out_put_dir):
        os.makedirs(images_out_put_dir)
    else:
        shutil.rmtree(images_out_put_dir)
        os.makedirs(images_out_put_dir) 


    if not os.path.exists(video_out_put_dir):
        os.makedirs(video_out_put_dir)
    else:
        shutil.rmtree(video_out_put_dir)
        os.makedirs(video_out_put_dir)


    ev_file = h5py.File(ev_file)
    event_frames = ev_file['data']

    bbox_color = (0, 255, 0)
    # bbox_color = (255, 255, 255)
    unknown_color = (0, 255, 255)

    pre_state = None

    for frame_index in tqdm(range(event_frames.shape[0])):
        # single_frame_shown = (event_frames[frame_index].sum(axis=0) * 60).astype(np.uint8)
        # single_frame_shown = cv2.cvtColor(single_frame_shown, cv2.COLOR_GRAY2BGR)

        ev_pr = event_frames[frame_index]
        num_bins = int(ev_pr.shape[0] / 2)
        height = int(ev_pr.shape[1])
        width = int(ev_pr.shape[2])
        ev_pr = np.transpose(ev_pr, (1, 2, 0))
        frame = np.zeros((height, width, 3), dtype=np.uint8) 

        for i in range(num_bins):
            pos_image = (ev_pr[:, :, i + num_bins]).astype(np.uint8)
            neg_image = (ev_pr[:, :, i]).astype(np.uint8)
            pos_image = cv2.equalizeHist(pos_image)
            neg_image = cv2.equalizeHist(neg_image)
            image = np.concatenate((neg_image[..., None], np.zeros((height, width, 1), dtype=np.uint8), pos_image[..., None]), axis=-1)
            frame = np.add(frame, image)  
        single_frame_shown = frame * 255.0

        event_frame = torch.tensor(event_frames[frame_index]).unsqueeze(0)
        event_frame = event_frame.to(gpus)

        width = event_frame.shape[3]
        height = event_frame.shape[2]

        if mode == 'gt':
            if frame_index in frame_to_ev:
                rgb_frame_index = int(np.where(frame_to_ev == frame_index)[0])
                # print('rgb_frame_index{}, len(frame_to_ev){}'.format(rgb_frame_index, len(frame_to_ev)))
                if rgb_frame_index+1 < len(frame_to_ev):
                    results = labels[frame_to_labels[rgb_frame_index]:frame_to_labels[rgb_frame_index+1]]
                else:
                    results = labels[frame_to_labels[rgb_frame_index]:]
            else:
                results = []


        elif mode == 'pre':
            with torch.inference_mode():
                results, pre_state = module.forward(event_frame, pre_state)
        if results is None:
            results = []
        if len(results) > 0: 
            for each_bbox in results:

                if mode == 'gt':
                    x1, y1 = list(each_bbox)[1:3]
                    w, h = list(each_bbox)[3:5]
                    x2, y2 = x1+w, y1+h
                    confidence = round(float(list(each_bbox)[6]), 2)
                    label_id = int(list(each_bbox)[5])

                    if label_id == 0 or label_id == 2:
                        color = bbox_color
                    else:
                        color = unknown_color

                elif mode == 'pre':
                    x1, y1, x2, y2 = list(each_bbox)[0:4]
                    confidence = round(float(list(each_bbox)[4]), 2)

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                abs_ =int(abs(y1-y2) * abs(x1-x2))
                if abs(y1-y2) * abs(x1-x2) < 200:
                    continue

                thickness = 2
                if mode == 'pre':
                    cv2.rectangle(single_frame_shown, (x1, y1), (x2, y2), bbox_color, thickness)
                    # cv2.putText(single_frame_shown, str(confidence), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, thickness)
                elif mode == 'gt':
                    cv2.rectangle(single_frame_shown, (x1, y1), (x2, y2), color, thickness)
                    # cv2.putText(single_frame_shown, str(confidence), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        output_file_path = images_out_put_dir + '/{}.png'.format(frame_index)

        cv2.imwrite(output_file_path, single_frame_shown)
    
    images = [img for img in os.listdir(images_out_put_dir) if img.endswith(".png")]
    images.sort(key=sort_key)
    video_name = video_out_put_dir + '/output_video.avi'
    fps = 60
    frame_size = (width, height) 
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, frame_size)  

    for image in images:
        image_path = os.path.join(images_out_put_dir, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    video_writer.release()
    print('Save the video in {}'.format(video_name))
    

    
if __name__ == '__main__':
    main()



