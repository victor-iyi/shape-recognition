"""Using image processing techniques to count shapes in images.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: heuristics.py
     Created on 26 June, 2018 @ 11:06 AM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

import cv2
import imutils
import numpy as np

from detector.base import Base


class Heuristics(Base):

    def __init__(self, **kwargs):
        super(Heuristics, self).__init__(**kwargs)

        # Sigma for lower & upper bound threshold.
        self._sigma = kwargs.get('sigma', 0.33)

        # Optimize performance.
        cv2.setUseOptimized(True)

    def _predict(self, image, **kwargs):
        # Calculate lower threshold and upper threshold using sigma = 0.33
        v = np.median(image)
        low = int(max(0, (1.0 - self._sigma) * v))
        high = int(min(255, (1.0 + self._sigma) * v))

        # Find edges in the image using canny edge detection method
        image = cv2.Canny(image, low, high)

        # Find contors.
        contors = cv2.findContours(image, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

        # Compatibility with OpenCV v2.x & v3.x
        contors = contors[0] if imutils.is_cv2() else contors[1]

        # Square, Circle & Triangle respectively
        shapes = [0, 0, 0]
        square, circle, triangle = 0, 0, 0

        # loop over the contours
        for contor in contors:
            # print(contor.shape)
            perimeter = cv2.arcLength(contor, True)

            # Apply contour approximation and store the result in vertices.
            vertices = cv2.approxPolyDP(contor, 0.04 * perimeter, True)

            square += int(len(vertices) == 4)
            circle += int(len(vertices) > 4)
            triangle += int(len(vertices) == 3)

        return square, circle, triangle
