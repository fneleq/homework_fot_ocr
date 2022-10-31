import os.path
from os.path import dirname, abspath

base_dir = dirname(dirname(dirname(abspath(__file__)))).replace('\\', '/')

icdar17_mlt_img_dir = os.path.join(base_dir, 'data/ctpn/train')
icdar17_mlt_gt_dir = os.path.join(base_dir, 'data/ctpn/train_gt')
pretrained_weight = os.path.join(base_dir, 'data/ctpn/checkpoints/v3_ctpn_ep22_0.3801_0.0971_0.4773.pth')

anchor_scale = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

IMAGE_MEAN = [123.68, 116.779, 103.939]
OHEM = True

checkpoints_dir = os.path.join(base_dir, 'data/ctpn/checkpoints')
outputs = os.path.join(base_dir, 'data/ctpn/logs')
