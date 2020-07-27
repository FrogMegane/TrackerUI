from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import time

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--save_video', default=True, type=bool,
                    help='save video with bounding box')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
            video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    checkpoint = torch.load(args.snapshot,
                            map_location=lambda storage, loc: storage.cpu())
    if 'epoch' in checkpoint:
         # if using self-training .pth
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    # For timing
    cnt = 0  # count for skip some frames in the begining of the video
    total_time = 0

    first_frame = True
    frame_list = []
    center_list = []
    init_bbx = []
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for frame in get_frames(args.video_name):
        if first_frame:
            frame_list.append(frame)
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
                init_bbx = init_rect
            except:
                exit()
            tracker.init(frame, init_rect)
            center = (init_rect[0]+init_rect[2]//2,
                      init_rect[1]+init_rect[3]//2)
            center_list.append(center)
            first_frame = False
        else:
            cnt += 1
            t1 = time.time()
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                polygon = polygon.reshape((-1, 1, 2))
                center = np.mean(polygon, axis=0)[0]  # get center
                center_list.append(center)
                cv2.polylines(frame, [polygon],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            frame_list.append(frame)
            cv2.imshow(video_name, frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                return
            t2 = time.time()
            total_time += (t2-t1)
    # Saving and print some information
    avg_fps = 1/total_time*cnt
    print('Avg fps: ', avg_fps)
    '''
    
    centers.reverse()#if needed
    '''
    if(args.save_video):
        print('Video Saving... ')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        size = (frame_list[0].shape[1], frame_list[0].shape[0])
        out = cv2.VideoWriter(
            './output/{:s}_out.mp4'.format(video_name), fourcc, int(avg_fps), size)
        # looping in the List of frames.
        for i in range(len(frame_list)):
            frame = frame_list[i]
            if i == 0:
                cv2.rectangle(frame, init_bbx, (255, 0, 0),
                              2)  # initial bounding boxes
            # only draw if 'polygon' in outputs:
            # for j in range(i+1):
            #    center=centers[j]
            #    draw_cross(frame,center)
            out.write(frame)
        fp = open("./output/{:s}.csv".format(video_name), "w")
        fp.write("frame,center_x,center_y\n")
        for i in range(len(center_list)):
            center = center_list[i]
            fp.write(str(i)+","+str(center[0])+","+str(center[1])+"\n")
        fp.close()


if __name__ == '__main__':
    main()
