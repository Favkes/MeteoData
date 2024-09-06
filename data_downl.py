"""
    This file contains functionality essential for the METEO project,
    as it is the essence of it's phase 1. (data extraction).
    It downloads, extracts and saves the data using the requests module
    as well as colour masking and kernels.
"""

import requests
from datetime import datetime
import cv2
import opencv_util as cvu
import numpy as np
import os
import warnings
from PIL import Image as PImage


class DataHandler:
    directory: str
    filename: str
    image: np.array
    edges: np.array
    n_frames: int

    def __init__(self, directory: str = ''):
        # Correct formatting precaution
        if directory:
            if not (directory.endswith('\\') or directory.endswith('/')):
                directory += '\\'

            # Directory exists precaution
            if os.path.exists(directory):
                _ = 'Directory already exists! May overwrite existing files.'
                warnings.warn(_)
            else:
                os.mkdir(directory)

        self.directory = directory

    @staticmethod
    def name_at_date(date: datetime, infix='.', infix2=None):
        """ Generates a filename in the format used in the project
                    for given datetime time value. """
        if infix2 is None:
            infix2 = infix
        date = (date.timestamp()//300-60)*300 + 18_000     # 60*5 = 300 and timezone correction
        date = datetime.fromtimestamp(date)
        # print(date)

        name = (f'{date.year}{infix}'
                f'{date.month}{infix}'
                f'{date.day}{infix2}'
                f'{date.hour}{infix}'
                f'{date.minute}')
        return name

    def update_name(self):
        """ Generates a filename in the format used in the project
            for the present time. """
        filename = self.name_at_date(
            datetime.now(),
            infix2='_'
        )
        self.filename = filename
        return filename

    def download_data(self, url: str = None, filename: str = None):
        """ Grabs the data from a URL using the requests module. """
        if filename is None:
            filename = self.filename
        if url is None:
            # url = 'https://meteo.org.pl/img/ra.png'
            url = 'https://meteo.org.pl/img/ra.gif'
        suffix = url[url.rfind('.'):]
        with open(f'{self.directory}{filename}{suffix}', 'wb') as file:
            try:
                data = requests.get(url).content
                file.write(
                    data
                )
            except Exception as exception:
                if type(exception).__name__ == "ConnectionError":
                    print('No connection to web.')
                    exit()
                assert type(exception).__name__ == 'NameError'
                assert exception.__class__.__name__ == 'NameError'
                assert exception.__class__.__qualname__ == 'NameError'
        gif = PImage.open(f'{self.directory}{filename}{suffix}')
        for index in range(gif.n_frames):
            # gif = gif.copy()
            gif.seek(index)
            # img.show()
            # img = np.array(img.convert('BGR'))
            gif.save(f'{self.directory}{filename}({index}).png')
        self.n_frames = gif.n_frames

    def extract_data(self, image: np.array, frame_index: int = 0, masked: bool = True):
        """ Extracts the data from an image using masking.
            Building in proper masking and formatting is advised. """
        # if image is None:
            # image = cv2.imread(f'{self.directory}{self.filename}.png')

        mask = [97, 148, 191, 255, 233, 249]
        output = cvu.mask_image(image, mask, return_mask=True)

        # THIS PIECE OF CODE TRIES TO REMOVE THE CUTTING OF CLOUDS BY CITY NAMES (TOP 5 EPIC FAIL)
        # mask = [60, 2, 251, 255, 255, 255]
        # city_names = cvu.mask_image(image, mask, return_mask=True)
        #
        # output = cv2.addWeighted(output, 1, city_names, 0.49, 0)

        # imgy, imgx = output.shape
        cv2.rectangle(output, (0, 639), (64, 575), (0, 0, 0), -1)

        if masked:
            cv2.imwrite(f'{self.directory}{self.filename}({frame_index}).png', output)
            self.image = output
            return output
        cv2.imwrite(f'{self.directory}{self.filename}({frame_index}).png', image)
        self.image = image
        return image

    def rechannel(self, n_channels=3):
        """ Converts single-channeled data image into an n-channeled image. """
        output = np.zeros((self.image.shape[0], self.image.shape[1], n_channels), np.uint8)
        for channel in range(n_channels):
            output[:, :, channel] = self.image[:, :]
        self.image = output
        return output

    @staticmethod
    def rechannel_image(image: np.array, n_channels=3):
        """ Converts single-channeled data image into an n-channeled image. """
        if len(image.shape) != 2:
            _ = 'The image provided is already multi-channeled, so this action is useless.'
            warnings.warn(_)
            return image
        output = np.zeros((image.shape[0], image.shape[1], n_channels), np.uint8)
        for channel in range(n_channels):
            output[:, :, channel] = image[:, :]
        return output

    def calculate_edges(self, image: np.array = None):
        """ This function is not efficient! If you NEED to find the edges,
            better rewrite it using the built-in kernels! """
        if image is None:
            image = cv2.imread(f'{self.directory}{self.filename}.png')

        processed = image.copy()
        for i in range(image.shape[0]*image.shape[1]):    # 640**2 = 409_600
            y, x = divmod(i, image.shape[1])
            # try:
            # if image.shape[0] == 4 and image.shape[1] == 3:
            #     print(i, y, x)
            kernel_h = cvu.kernel_horizontal(image, (x, y))
            kernel_v = cvu.kernel_vertical(image, (x, y))
            # except IndexError:
            #     print(x, y, image.shape)
            #     exit()
            # print(kernel_v[0], kernel_h[0])
            # we access any of the 3 channels as they carry the same values
            if kernel_h[0][0] != kernel_h[1][0] or kernel_v[0][0] != kernel_v[1][0]:
                processed[y, x] = 255
            else:
                processed[y, x] = 0

        self.edges = processed
        return processed

    def convert_frames(self):
        """
            Overrides existing frames by their masked representations.
        :return:
        """
        for index in range(self.n_frames):
            # image = cv2.imread(f'{self.directory}pablo.png')    #{self.directory}{self.filename}({index})
            image = cv2.imread(f'{self.directory}{self.filename}({index}).png')  # {self.directory}{self.filename}({index})
            self.extract_data(image, frame_index=index, masked=True)
        # image = self.rechannel_image(image)


if __name__ == '__main__':
    warnings.warn('This file is a module with nothing to run. Art thou lost, traveler?')
