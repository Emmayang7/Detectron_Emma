#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
    (e.g., .jpg) in a folder.
    """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

import numpy as np #Yunhan Yang

def video_frame(args):  #Yunhan Yang
    print(cv2.__version__)
    vidcap = cv2.VideoCapture(args.input_dir)
    success,image = vidcap.read()
    count = 0
    success = True
    total = 0
    while success:  #do two frame for testing
        success,image = vidcap.read()
        #print 'Read a new frame: ', success
        #image now is matrix
        if count % args.process_rate == 0:
            cv2.imwrite(args.im_or_folder+ "frame%d.jpg" % total, image)  # save frame as JPEG file in a folder
                total += 1
        count += 1
    return total-1


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
                        '--cfg',
                        dest='cfg',
                        help='cfg model file (/path/to/model_config.yaml)',
                        default=None,
                        type=str
                        )
    parser.add_argument(
                        '--wts',
                        dest='weights',
                        help='weights model file (/path/to/model_weights.pkl)',
                        default=None,
                        type=str
                        )
    parser.add_argument(
                        '--image-ext',
                        dest='image_ext',
                        help='image file name extension (default: jpg)',
                        default='jpg',
                        type=str
                        )
                                            
    parser.add_argument(
                        'im_or_folder', help='image or folder of images', default=None
                        )
    #Yunhan Yang
    parser.add_argument(
                        '--src_video_path',
                        dest='input_dir',
                        help='directory for input video (default: /tmp/infer_simple)',
                        default='/tmp/infer_simple',
                        type=str
                        )
    parser.add_argument(
                        '--output_format',
                        dest='video_format',
                        help='choose the format for your video',
                        default='output.mp4',
                        type=str
                        )
    parser.add_argument(
                        '--resize_ratio',
                        dest='video_resize',
                        help='resize each frame',
                        default='0.5',
                        type=float
                        )
    #The solution is to first resize your images such that the short side is around 600-800px (the exact choice does not matter) and then run inference on the resized image.
    parser.add_argument(
                        '--processing_step',
                        dest= 'process_rate',
                        help='if this value = 5, Detectron will process one out of five frames',
                        default='5',
                        type=int
                        )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    
    video_frame_length = video_frame(args)  #Yunhan Yang pass in a list to save all the frames
    print ("The video has length " + str(video_frame_length) + " frames")
    
    
    f_image_path = args.im_or_folder+ "frame%d.jpg" % 0
    fr = cv2.imread(f_image_path, 0)
    origin_width, origin_height = fr.shape[:2]
    
    for x in range(0,video_frame_length):
        im_list = [args.im_or_folder+"frame" + str(x) + ".jpg"] #Yunhan Yang have to save frame in real folder and then read in

        #maybe need need double for loop for list of frames
        for i, im_name in enumerate(im_list):
            out_name = os.path.join(
                                    args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
                                    )
            logger.info('Processing {} -> {}'.format(im_name, out_name))
            im = cv2.imread(im_name)
            timers = defaultdict(Timer)
            t = time.time()
            with c2_utils.NamedCudaScope(0):
                cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                                                                             model, im, None, timers=timers
                                                                             )
            logger.info('Inference time: {:.3f}s'.format(time.time() - t))
                for k, v in timers.items():
                    logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
                        if i == 0:
                            logger.info(
                                        ' \ Note: inference on the first image will be slower than the '
                                        'rest (caches and auto-tuning need to warm up)'
                                        )
            
            #Yunhan Yang edit
            if not os.path.exists(args.im_or_folder+ "/video"):
                os.makedirs(args.im_or_folder+ "/video")
            #Yunhan Yang edit Detectron/lib/utils/vis.py add make result as jpg than pdf

            vis_utils.vis_one_image(
                                    im[:, :, ::-1],  # BGR -> RGB for visualization
                                    im_name,
                                    args.output_dir,
                                    cls_boxes,
                                    cls_segms,
                                    cls_keyps,
                                    dataset=dummy_coco_dataset,
                                    box_alpha=0.3,
                                    show_class=True,
                                    thresh=0.7,
                                    kp_thresh=2
                                    )

    dir_path = args.im_or_folder+ "/video"
    out_path = "/tmp/detectron-video/"
    ext = args.image_ext
    output = args.video_format

    images = []
    count =0
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)

    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    ##cv2.imshow('video',frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    #out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))


    out = cv2.VideoWriter(output,cv2.VideoWriter_fourcc('m','p','4','v'), args.output_fps, (int(args.video_resize*origin_width), int(args.video_resize*origin_height)))

    for image in images:
    
        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)
        res = cv2.resize(frame,(int(args.video_resize*origin_width), int(args.video_resize*origin_height)), interpolation = cv2.INTER_CUBIC)  #Yunhan Yang resize the image
        out.write(res) # Write out frame to video
    
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break


    print("The output video is {}".format(output))


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
