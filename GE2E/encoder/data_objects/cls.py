from encoder.data_objects.random_cycler import RandomCycler
from pathlib import Path
import cv2
import numpy as np

class Cls:
    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        self.images = [image for image in self.root.iterdir()]
        self.image_cycler = RandomCycler(self.images)

    def random_sample(self, count):
        images = self.image_cycler.sample(count)
        return [self.process_img(cv2.imread(str(img)), 112)/255. for img in images]

    def process_img(self, img, size):

        height, width, channels = img.shape
        if width == height:
            return np.transpose(cv2.resize(img, (size, size)), (2,0,1))

        if width > height:
            new_size = width
        else:
            new_size = height

        new_x = (new_size - width) // 2
        new_y = (new_size - height) // 2
        new_img = cv2.copyMakeBorder(img, new_y, new_y+new_size, new_x, new_x+new_size, cv2.BORDER_CONSTANT, value=[0,0,0])
        return np.transpose(cv2.resize(new_img, (size, size)), (2,0,1))
   