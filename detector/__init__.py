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
     File: __init__.py
     Created on 26 June, 2018 @ 11:06 AM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
from detector.heuristics import Heuristics
from detector.network import Network

__all__ = [
    'heuristics',
    'network',
]
