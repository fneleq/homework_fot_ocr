import os.path
from os.path import dirname, abspath

base_dir = dirname(dirname(abspath(__file__)))
checkpoints = os.path.join(base_dir, 'data/ctpn/checkpoints')
images = os.path.join(base_dir, 'data/ctpn/train/image_5.jpg')
anchor_scale = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

IMAGE_MEAN = [123.68, 116.779, 103.939]
OHEM = False
