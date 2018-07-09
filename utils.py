"""Shape detector utility file.

   @description
     Utility file for loading, processing, cleaning and manipulating
     dataset. It also provides functional APIs for creating Sequence
     generators.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: detector.py
     Created on 26 June, 2018 @ 11:12 AM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import glob

import numpy as np
from keras.utils import Sequence

import cv2
import detector


def process_txt(label_path: str):
    """Read label & split shapes appropriately.

    Arguments:
        label_path {str} -- File path to label.

    Returns:
        tuple -- Splitted shapes: square, circle, triangle.
    """

    # Read, strip white space & split label.
    with open(label_path, mode='r') as f:
        labels = str.strip(f.read()).split(', ')

    # Square, Circle, Triangle.
    shapes = map(lambda l: int(l.split(':')[1]), labels)

    return tuple(shapes)


def process_img(img_path: str, size: tuple=(200, 200), sigma=0.33):
    """Read image file & resize image.

    Arguments:
        img_path {str} -- Image file path.

    Keyword Arguments:
        size {tuple} -- Size of resized image. (default: {(200, 200)})

    Returns:
        array-like -- Array of image (width, height, channel)
    """
    # Read image as BGR.
    img = cv2.imread(img_path)

    # Blur image (for noise reduction).
    # kernel = np.ones((5, 5), np.float32) / 25
    # img = cv2.filter2D(img, -1, kernel)
    # img = cv2.medianBlur(img, 5)

    # Convert to Gray image.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = cv2.Laplacian(img, cv2.CV_64F)

    # Resize image.
    img = cv2.resize(img, (200, 200))
    # Otsu's thresholding after Gaussian filtering
    img = cv2.GaussianBlur(img, (5, 5), 0)

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY, 11, 2)
    return thresh


def load_data(data_dir: str='training_data'):
    """Functional generator API for loading training data.

    Keyword Arguments:
        data_dir {str} -- Data file path. (default: {'training_data'})

    Yields:
        {tuple} -- tuple of array-like elements containing
            pre-processed images & labels.
    """
    img_paths, txt_paths = get_files(data_dir=data_dir)

    for img_path, txt_path in zip(img_paths, txt_paths):
        # Process image files.
        image = process_img(img_path)
        label = process_txt(txt_path)

        yield image, label


def get_files(data_dir: str='training_data'):
    """Retrieve each image & label file paths.

    Keyword Arguments:
        data_dir {str} -- Path to directory containing images & labels. (default: {'training_data'})

    Returns:
        {tuple} -- image file paths and label file paths.
    """

    # Image & text files.
    imgs = glob.glob(data_dir + '/*.jpg')
    txts = glob.glob(data_dir + '/*.txt')

    assert len(imgs) == len(txts), 'Labels & images must have same length.'

    return imgs, txts


class DataGenerator(Sequence):

    def __init__(self, x_path, y_path, batch_size):
        self.x, self.y = x_path, y_path
        self.batch_size = batch_size

    def __len__(self):
        """Return data length.

        Returns:
            {int} -- Dataset length.
        """

        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        """Get an item by index.

        Arguments:
            idx {int} -- Element index.

        Returns:
            {tuple} -- tuple of array-like elements containing
                pre-processed images & labels.
        """

        # Split batches.
        x_files = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        y_files = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        # Pre-process individual entry's image & label.
        batch_x, batch_y = [], []
        for x_file, y_file in zip(x_files, y_files):
            # Read images.
            batch_x.append(process_img(x_file))
            batch_y.append(process_txt(y_file))

        # Convert batches to Numpy arrays.
        batch_x = np.asarray(batch_x, dtype=np.float32)
        batch_y = np.asarray(batch_y, dtype=np.int64)

        return batch_x, batch_y

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass


def main():
    data = load_data()
    net = detector.Heuristics()
    score = net.score(data)
    print('\nScore: {:.2%}'.format(score))

    # for i, (image, label) in enumerate(data):
    #     if i > 3:
    #         break
    #     image = cv2.Laplacian(image, cv2.CV_64F)

    #     cv2.imshow(str(label), image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
