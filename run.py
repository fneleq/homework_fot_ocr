import os
import numpy as np
from ocr import ocr
import time
import shutil
from PIL import Image
from glob import glob
import argparse
import os.path
from os.path import dirname, abspath
base_dir = dirname(abspath(__file__))
test_path = os.path.join(base_dir, 'data/test_images/*.*')
result_path = os.path.join(base_dir, 'data/result')
paser = argparse.ArgumentParser('Chinese OCR')
paser.add_argument('--input', default=test_path, type=str, help='the directory of input images')
paser.add_argument('--output', default=result_path, type=str, help='the directory of output file')



def single_pic_proc(image_file):
    image = np.array(Image.open(image_file).convert('RGB'))
    result, image_framed = ocr(image)
    return result, image_framed


if __name__ == '__main__':
    args = paser.parse_args()
    image_files = glob(args.input)
    result_dir = args.output

    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    for image_file in sorted(image_files):
        t = time.time()
        result, image_framed = single_pic_proc(image_file)
        output_file = os.path.join(result_dir, image_file.split('\\')[-1])
        text_file = os.path.join(result_dir, image_file.split('\\')[-1].split('.')[0] + '.txt')
        txt_f = open(text_file, 'w', encoding='utf-8')
        Image.fromarray(image_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        for key in result:
            txt_f.write(result[key][1] + '\n')
        txt_f.close()
