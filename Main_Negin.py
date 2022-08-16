from utils import get_dataset, int64_feature, int64_list_feature, bytes_feature, bytes_list_feature, float_list_feature
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow.compat.v1 as tf
import os
from matplotlib.lines import Line2D
import io
import IPython.display as display
from PIL import Image
from waymo_open_dataset import dataset_pb2 as open_dataset



def display_image(batch):
    # color map classes
    colormap = {1: [1, 0, 0], 2: [0, 1, 0], 4: [0, 0, 1]}
    # matching legend lines
    legend_lines = [Line2D([0], [0], color=colormap[1], lw=1, label="vehicles"), 
                    Line2D([0], [0], color=colormap[2], lw=1, label="pedestrians"), 
                    Line2D([0], [0], color=colormap[4], lw=1, label="bicycles")]
    
    # number of rows in plot
    num_rows = 2
    
    # define rows/cols in figure
    if num_images % num_rows == 0:
        num_cols = num_images // num_rows
    else:
        num_cols = num_images % num_rows + 1
        
    # set up figure
    f, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))
    f.suptitle('Explore GroundTruth Data')
    
        ### for each tfrecord in batch...
    for idx, record in enumerate(batch):

        # getting specs
        img = record["image"].numpy()
        img_shape = record["original_image_spatial_shape"].numpy()
        filename = record["filename"]
        bboxes = record["groundtruth_boxes"].numpy()
        classes = record["groundtruth_classes"].numpy()

        # getting image and put in background
        curr_row = idx % num_rows
        curr_col = idx % num_cols
        axs[curr_row, curr_col].imshow(img)

        # adding bounding boxes to the foregound
        for box, cl in zip(bboxes, classes):
            # get box coordinates
            y1, x1, y2, x2 = box
            # rescale to image size
            x1, x2 = img_shape[0]*x1, img_shape[0]*x2
            y1, y2 = img_shape[1]*y1, img_shape[1]*y2
            # define rectangle and color with colormap
            rec = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none', edgecolor=colormap[cl])
            # add rectangle to plot
            axs[curr_row, curr_col].add_patch(rec)
            
    # adding legend and adjust layout
    axs[num_rows-1,0].legend(handles=legend_lines, loc='center', bbox_to_anchor=(0,-0.2))
    plt.tight_layout()
    plt.show()

    # save figure to png
    f.savefig('Results_Negin/data_exploratory_analysis.png')
    pass
    
def process_tfr(dataset_tf, path_tf):
    """
    process a tf record (selected from training folder) into a tf api tf record
    """

    # display dataset specs
    print('#########################################TFrecord Information#########################################')
    print(dataset_tf.element_spec)
    
    ### create batch for display
    # batch size
    global num_images
    num_images = 10
    # Number of filenames to read in shuffle
    num_files = len([entry for entry in os.listdir(path_tf) if os.path.isfile(os.path.join(path_tf, entry))]) #86
   #Select random tfrecord files
    batch_tf = dataset_tf.shuffle(num_files, reshuffle_each_iteration=True).take(num_images)
    
#     # display image size 
#     for record in batch_tf:
#         print('#################Image Size#################',record["original_image_spatial_shape"].numpy())
#      # bboxes in format [0...1] - rescale to image size
#     for record in batch_tf:
#         print('#################GrndTruth BBX#################', record["groundtruth_boxes"].numpy())
    
    
    # STEP1(EDA)- Display random images from train dataset: num_images
    display_image(batch_tf)
    print('#########################################STEP1 (EDA)- Image/BBX Display Completed for the Batch#########################################')
    
    ## STEP2- Edit Config File- done in terminal
#     cd /home/workspace/experiments/pretrained_model/
#     wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
#     tar -xvzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
#     rm -rf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
#     cd /home/workspace/
#     python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
#     mv pipeline_new.config /home/workspace/experiments/reference/

    ## STEP3- Model Training & Evaluation
## Training Process
#   python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
## Tensorflow board
#   python -m tensorboard.main --logdir experiments/reference/
## Evaluation Process
# python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/

    ## STEP4- Improve Performance    
#     python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference --output_directory experiments/reference/exported/
    
    
if __name__=="__main__":   
#     path_tfrecord= os.path.join(os.getcwd(),'data','train' ,'segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord')
#     dataset_tf = get_dataset(path_tfrecord,)
    path_tfrecord= os.path.join(os.getcwd(),'data','train/') #"/home/workspace/data/train/"
    dataset_tf = get_dataset(f"{path_tfrecord}*.tfrecord",)
    process_tfr(dataset_tf, path_tfrecord)
