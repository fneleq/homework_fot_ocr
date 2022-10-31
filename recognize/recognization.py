import os.path

import torch
from torch.autograd import Variable
from PIL import Image
import cv2
import recognize.config as config
from recognize.model import CRNN
from recognize.utils import strLabelConverter, resizeNormalize


class Ocr():
    def __init__(self, model_path):
        self.alphabet = config.alphabet
        self.nclass = config.nclass
        self.model = CRNN(config.imgH, config.nc, self.nclass, 256)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.converter = strLabelConverter(self.alphabet)

    def predict(self, img):
        h, w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(img)
        transform = resizeNormalize((int(w / h * 32), 32))
        img = transform(img)
        img = img.view(1, *img.size())
        img = Variable(img)
        img = img.cuda()
        preds = self.model(img)
        _, preds = preds.max(2)

        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        text = self.converter.decode(preds.data, preds_size.data, raw=False).strip()
        return text


if __name__ == '__main__':
    model_path = config.pretrained_model_path
    recognizer = Ocr(model_path)
    image = os.path.join(config.image_dir, '0.png')
    image = cv2.imread(image)
    res = recognizer.predict(image)
    print(res)
