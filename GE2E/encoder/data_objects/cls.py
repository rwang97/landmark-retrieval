from encoder.data_objects.random_cycler import RandomCycler
from pathlib import Path
import cv2
import random
import numpy as np

class Cls:
    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        self.images = [image for image in self.root.iterdir()]
        self.image_cycler = RandomCycler(self.images)

    def random_sample(self, count):
        images = self.image_cycler.sample(count)
        return [self.process_img(cv2.imread(str(img)), 224)/255. for img in images]

    # Randomly change the brightness of imgs
    def random_brightness(self, img, max_change=60):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # generate random brightness noise
        value = random.randint(-max_change, max_change+1)

        # Clap the result to 0 - 255
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            value = int(-value)
            lim = 0 + value
            v[v < lim] = 0
            v[v >= lim] -= value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def process_img(self, img, size):

        height, width, channels = img.shape
        if width == height:
            new_img = cv2.resize(img, (size, size))
        else:
            ratio = float(size)/max(height, width)
            new_height = int(height*ratio)
            new_width = int(width*ratio)

            new_img = cv2.resize(img, (new_width, new_height))

            delta_w = size - new_width
            delta_h = size - new_height
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            new_img = cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

        p_flip = 0.5
        if (random.uniform(0,1) > 1 - p_flip):
            new_img= cv2.flip(new_img, 1)

        p_brightness = 0.8
        if (random.uniform(0,1) > 1 - p_flip):
            new_img = self.random_brightness(new_img)

        # print(img.shape)
        # print(new_img.shape)
        # cv2.imwrite("/datadrive/google-landmark/landmark-retrieval/ArcFace/augmented/original.jpg", img)
        # cv2.imwrite("/datadrive/google-landmark/landmark-retrieval/ArcFace/augmented/augmented.jpg", new_img)
        # exit(0)

        return np.transpose(new_img, (2, 0, 1))
