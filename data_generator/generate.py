import argparse
import os
import random
from os.path import dirname, abspath

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

base_dir = dirname(dirname(abspath(__file__)))
base_dir = base_dir.replace('\\', '/')
info_path = os.path.join(base_dir, 'data/generator/info.txt')
file_path = os.path.join(base_dir, 'data/crnn_dataset/validation.txt')
image_path = os.path.join(base_dir, 'data/crnn_dataset/val_images')
background = os.path.join(base_dir, 'data/generator/background')
font_path = os.path.join(base_dir, 'data/generator/font')
parser = argparse.ArgumentParser('Chinese Test Pic Generator')
parser.add_argument('--len', default=10, type=int, help='the length of generated text pic')
parser.add_argument('--num', default=1000, type=int, help='the number of pic wanna generate')
parser.add_argument('--infopath', default=file_path, type=str, help='the path to save info file')
parser.add_argument('--imagepath', default=image_path, type=str, help='the path to save image file')
parser.add_argument('--background', default=background, type=str, help='the background images used to generate image')
parser.add_argument('--fontpath', default=font_path, type=str, help='the font used to generate image')


def text_selection(num):
    start = random.randint(0, len(info_str) - num - 1)
    end = start + num
    words = info_str[start:end]
    return words


def image_gen(background, width, height):
    background_list = os.listdir(background)
    background_choice = random.choice(background_list)
    bg = Image.open(os.path.join(background, background_choice))
    x, y = random.randint(0, bg.size[0] - width), random.randint(0, bg.size[1] - height)
    bg = bg.crop((x, y, x + width, y + height))
    return bg


def random_font_size():
    font_size = random.randint(24, 27)
    return font_size


def random_word_color():
    font_color_choice = [[54, 54, 54], [54, 54, 54], [105, 105, 105]]
    font_color = random.choice(font_color_choice)
    noise = np.array([random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)])
    font_color = (np.array(font_color) + noise).tolist()
    return tuple(font_color)


def random_font(fontpath):
    font_list = os.listdir(fontpath)
    select_font = random.choice(font_list)
    return os.path.join(fontpath, select_font)


def randon_x_y(size, font_size):
    width, height = size
    x = random.randint(0, width - font_size * 10)
    y = random.randint(0, int((height - font_size) / 4))
    return x, y


def darken(image):
    filter_ = random.choice([
        ImageFilter.SMOOTH,
        ImageFilter.SMOOTH_MORE,
        ImageFilter.GaussianBlur(radius=1.3)
    ])
    image = image.filter(filter_)
    return image


def gen(save_path, num, file):
    words = text_selection(args.len)

    raw_image = image_gen(args.background, 280, 32)

    font_size = random_font_size()

    font_name = random_font(args.fontpath)

    font_color = random_word_color()

    x, y = randon_x_y(raw_image.size, font_size)

    font = ImageFont.truetype(font_name, font_size)
    draw = ImageDraw.Draw(raw_image)
    draw.text((x, y), words, fill=font_color, font=font)

    raw_image = darken(raw_image)

    image_name = str(num) + '.png'

    file.write(image_name + '\t' + words + '\n')
    raw_image.save(os.path.join(save_path, image_name))


if __name__ == '__main__':
    args = parser.parse_args()
    with open(info_path, 'r', encoding='utf-8') as f:
        info_list = [part.strip().replace('\t', '') for part in f.readlines()]
        info_str = ''.join(info_list)

    file = open(args.infopath, 'w', encoding='utf-8')

    for i in range(args.num):
        gen(args.imagepath, i, file)
        if i % 50 == 0:
            print(f'[{i}/{args.num}]')

    file.close()
