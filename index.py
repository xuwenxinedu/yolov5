# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys


import torch
import torch.backends.cudnn as cudnn


from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
ROOT = '.'
# 0: barrier 1:coast
@torch.no_grad()
def run(
        weights='./best.pt',  # model.pt path(s)
        source='./ddirs',  # file/dir/URL/glob, 0 for webcam
        data='./data/coco128.yaml',  # dataset.yaml path
        imgsz=(1920, 1080),  # inference size (height, width)
        conf_thres=0.35,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    num2cls = {0: 'barrier', 1: 'coast'}
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    all_objs = []
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        # Inference
        pred = model(im, augment=augment, visualize=False)
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        for i, det in enumerate(pred):  # per image
            p, im0 = path, im0s.copy()       
            
            if len(det) > 0:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = num2cls[int(cls)]
                    img_id = p[:-4]
                    confidence = float(conf)
                    xmin = int(xyxy[0]) if int(xyxy[0]) > 0 else 0
                    ymin = int(xyxy[1]) if int(xyxy[1]) > 0 else 0
                    xmax = int(xyxy[2]) if int(xyxy[2]) < 1920 else 1920
                    ymax = int(xyxy[3]) if int(xyxy[3]) < 1080 else 1080
                    all_objs.append([label, img_id, confidence, 
                                    xmin, ymin, xmax, ymax])
            else:
                all_objs.append(['', p[:-4],'','','','',''])
    print(all_objs)
    return all_objs
            
def vid2img(file_path) :
    if not os.path.exists('./ddirs/'):
        os.makedirs('./ddirs')
    for file in os.listdir(file_path):
        path = os.path.join(file_path, file)
        if path[-4:] != '.mp4':
            continue
        vidcap = cv2.VideoCapture(path)
        if not vidcap.isOpened():
            return False
        success,image1 = vidcap.read()
        image2 = image1
        while success:
            image2 = image1
            success, image1 = vidcap.read()
        pic_name = os.path.basename(path)[:-4] + ".jpg"
        c = os.path.join("./ddirs/", pic_name)
        cv2.imwrite(c, image2)
    return True

def main():
    check_requirements(exclude=('tensorboard', 'thop'))
    run()

def invoke(_input:str) :
    vid2img(_input)
    main()

if __name__ == "__main__":
    invoke('')
