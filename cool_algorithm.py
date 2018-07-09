"""Recognize and count the number of three different shapes.

   @description
     The goal is to implement an algorithm to recognize and
     count the number of occurrence of three different shapes
     (i.e. square, circle, triangle) in a given image.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: detector.py
     Created on 27 June, 2018 @ 12:57 AM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import argparse

import utils
# Helper file.
from detector import Heuristics, Network


def main(args):

    # Create Network instance.
    model = args.method()

    filenames = utils.get_files(args.img_dir) if args.img_path is None else [
        args.img_path]

    if args.mode.lower() == "predict":
        # Make predictions.
        prediction = model.predict(filenames=filenames)
        prediction = list(map(int, prediction[0]))

        # Pretty-print predictions.
        print('\n{0}\n{1:^65}\n{0}'.format('=' * 65, "PREDICTIONS"))
        print('Squares: {}\tCircles: {}\tRectangles: {}'.format(*prediction))
        print('{}'.format('-' * 65))
    else:
        # Get true labels & images.
        prediction = model.score()


def str_to_list(str_paths: str):
    str_paths.replace(", ", ",")

    return str_paths.split(',') or str_paths.split(', ')


def meth(method: str):
    # Convert method to proper class font-style.
    studlyCase = method.strip().title().replace(" ", "")
    # print(studlyCase)
    # Return class fro string. e.g. `method = Heuristics`
    if method.lower() == "network":
        return Network

    return Heuristics


if __name__ == '__main__':
    # Command line argument parser.
    parser = argparse.ArgumentParser(
        description='Recognizing & counting number of shape occurrences.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional Arguments. (for a single image)
    parser.add_argument('--img_path', type=str, default=None,
                        help='Path to a single image file to be used for prediction.')

    parser.add_argument('--img_dir', type=str, default="training_data",
                        help="In case there are multiple images, specify it's directory here.")

    parser.add_argument('--method', type=meth, default="heuristics",
                        help="Method to use for prediction. One of 'heuristics' or 'network'")

    parser.add_argument('--mode', type=str, default="score",
                        help="Predict or evaluate accuracy. One of 'score' or 'predict'")
    # Optional arguments.
    parser.add_argument('-m', '--model_path', type=str,
                        default='saved/models/model.interrupt.h5',
                        help='For `Network`: Path to a saved `.h5` Keras Model.')

    # Parse command line arguments.
    args = parser.parse_args()

    # When in `mode` == 'score'... Only allow `img_dir`

    # Start the program.
    main(args)
