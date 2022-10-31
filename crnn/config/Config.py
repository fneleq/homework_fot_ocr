import os.path
import pickle as pkl
from os.path import dirname, abspath


def load_alphabet():
    file_path = os.path.join(base_dir, 'data/crnn_dataset/alphabet.pkl')
    alphanet_list = pkl.load(open(file_path, 'rb'))
    alphabet = [ord(ch) for ch in alphanet_list]
    return alphabet


#####################
# random parameters #
#####################
random_seed = 42
random_sample = True

#################
# io parameters #
#################
base_dir = dirname(dirname(dirname(abspath(__file__)))).replace('\\', '/')
image_dir = os.path.join(base_dir, 'data/crnn_dataset/images')
eval_image_dir = os.path.join(base_dir, 'data/crnn_dataset/val_images')
info_dir = os.path.join(base_dir, 'data/crnn_dataset/data.txt')
eval_dir = os.path.join(base_dir, 'data/crnn_dataset/validation.txt')
save_path = os.path.join(base_dir, 'data/crnn_dataset/checkpoints')
pretrained_model = 'CRNN-1010.pth'
pretrained_model_path = os.path.join(save_path, pretrained_model)

####################
# model parameters #
####################
alphabet = load_alphabet()
imgH = 32
imgW = 280
nc = 1
nclass = len(alphabet) + 1
nh = 256
batch_size = 50
lr = 1e-3
adam = True
adadelta = False
keep_ratio = True
beta = 0.5
epochs = 1
