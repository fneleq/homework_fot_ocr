import PIL
import cv2
import numpy as np
from PIL import Image
from PIL import Image


def getpilimage(image):
    if isinstance(image, PIL.Image.Image):  # or isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
        return image
    elif isinstance(image, np.ndarray):
        return cv2pil(image)


def getcvimage(image):
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, PIL.Image.Image):  # or isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
        return pil2cv(image)


def pil2cv(image):
    if len(image.split()) == 1:
        return np.asarray(image)
    elif len(image.split()) == 3:
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    elif len(image.split()) == 4:
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGR)


def cv2pil(image):
    assert isinstance(image, np.ndarray), 'input image type is not cv2'
    if len(image.shape) == 2:
        return Image.fromarray(image)
    elif len(image.shape) == 3:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
