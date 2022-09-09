# ================================================================
#
#   File name   : detection_demo.py
#   Author      : PyLessons
#   Created date: 2020-09-27
#   Refactor    : Ucok23
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
# ================================================================
import os

from yolo.configs import *
from yolo.utils import detect_image, load_yolo_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

image_path = "./IMAGES/kite.jpg"
video_path = "./IMAGES/test.mp4"

yolo = load_yolo_model()
detect_image(yolo, image_path, "./IMAGES/kite_pred.jpg", input_size=YOLO_INPUT_SIZE, show=True,
             rectangle_colors=(255, 0, 0))

# detect_video(yolo, video_path, "", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))

# detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255, 0, 0))

# detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=False, \
# rectangle_colors=(255,0,0), realtime=False)
