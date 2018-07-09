"""Shape detector utility file.

   @description
     Using Neural Network for shape detection. Recorded accuracy isn't
     impressive here. However, shape detection in this case is best solved
     with image processing heuristics like contour findings, edge detection &
     line tracking.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: detector.py
     Created on 26 June, 2018 @ 11:06 AM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import utils
import cv2
import numpy as np
import imutils


def main():
    data = utils.load_data('training_data')
    for i, (image, label) in enumerate(data):
        if i < 1:
            continue
        if i > 5:
            break

        # Find edges in the image using canny edge detection method
        # Calculate lower threshold and upper threshold using sigma = 0.33
        sigma, v = 0.33, np.median(image)
        low = int(max(0, (1.0 - sigma) * v))
        high = int(min(255, (1.0 + sigma) * v))

        edged = cv2.Canny(image, low, high)

        contors = cv2.findContours(edged, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

        # Compatibility with OpenCV v2.x & v3.x
        contors = contors[0] if imutils.is_cv2() else contors[1]

        # Now we will loop over every contour
        # call detectShape() for it and
        # write the name of shape in the center of image
        # Square, Circle & Triangle respectively
        shapes = [0, 0, 0]

        # shapes = {
        #     "Square": 0,
        #     "Circle": 0,
        #     "Triangle": 0,
        # }
        # print(np.squeeze(contors[0], axis=1).shape)
        # print(np.squeeze(contors[0], axis=1))
        # break

        # loop over the contours
        for contor in contors:
            # print(contor.shape)
            perimeter = cv2.arcLength(contor, True)

            # apply contour approximation and store the result in vertices
            vertices = cv2.approxPolyDP(contor, 0.04 * perimeter, True)
            n_vertices = len(vertices)
            # print(vertices.shape)

            if len(vertices) == 3:
                shapes[2] += 1  # Triangle
            elif len(vertices) == 4:
                shapes[0] += 1  # Square
            else:
                shapes[1] += 1  # Circle

        print(shapes, label)
        # break


if __name__ == '__main__':
    main()
