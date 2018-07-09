"""Base class for detectors.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: base.py
     Created on 26 June, 2018 @ 11:06 AM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
from abc import ABCMeta, abstractmethod

import numpy as np

import utils


class Base(object):

    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self._verbose = kwargs.get('sigma', 1)

    def __call__(self, filenames=None, **kwargs):
        """Make object callable. Same signature as `self.predict`.

        See `self.predict` for details.
        """
        return self.predict(filenames=filenames, **kwargs)

    def __repr__(self):
        return 'base.Base(verbose={})'.format(self._verbose)

    def __str__(self):
        return self.__repr__()

    def predict(self, filenames=None, images=None, **kwargs):
        """Make predictions given image file paths.

        Arguments:
            filenames {tuple} -- Iterable containing file names or a generator
                that yields `filenames, labels`.

        Returns:
            {np.ndarray} -- array-like contianing predicted values.
        """

        if images is None and filenames is not None:
            # Convert filenames to images.
            images = (utils.process_img(file) for file in filenames)
        elif filenames is None and images is not None:
            pass
        else:
            raise ValueError('Supply either `filenames` or `images`.')

        # Make predictions on each image.
        prediction = [self._predict(im, **kwargs) for im in images]

        return np.asarray(prediction)

    def score(self, images_or_gen, labels=None):
        data = images_or_gen if labels is None else zip(images_or_gen, labels)
        correct, total = 0, 0

        for image, label in data:
            shape = self._predict(image)
            correct += int(shape == label)
            total += 1

            if self._verbose == 1:
                print('\rCorrect: {:,} of {:,}'.format(correct, total),
                      end='')

        return correct / total

    @abstractmethod
    def _predict(self, X, **kwargs):
        return NotImplemented

    def _train(self, X, y, **kwargs):
        return NotImplemented
