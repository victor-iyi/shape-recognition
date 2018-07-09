# Shape Recognition

> *Demo of how Neural Networks can fail you!*

Using image processing techniques (heuristics) or using Network to count how many shapes are there in an image. Which will give better performance? Or perhaps if both works together?!

Here, I demonstrated how neural networks isn't the answer to many software challenges (*especially in Computer Vision*).

## Results

Neural network accuracy: **0.00%**

Heuristics accuracy: **49.36%**

## Usage

### Cloning the repository

Clone or download this repository.

```sh
> git clone https://github.com/victor-iyiola/shape-recognition.git
> cd shape-recognition
```

### Getting the Dataset

Dataset has been nicely packaged for you [here](https://dropbox.com). Copy the `training_data` into the root directory.

```sh
> tree -L 1
.
├── README.md
├── cool_algorithm.py
├── detector
├── preprocessing.py
├── training_data
└── utils.py
```

### Command line arguments

To see available command line arguments simply use the `--help` flag.

```sh
> python cool_algorithm.py --help
```

### Run with Image Processing Heuristics

The `--method` flag specifies which mode to run the script. `heuristics` signifies Image processing heuristics method which `network` uses Neural Network.

```sh
> python cool_algorithm.py --mode=score --method=heuristics
```

## Road map

- Using `cv2.findContours` to retrieve possible area containing a certain shape. Feed those contours into a network that predicts the probability of this contour containing a *triangle*, *rectangle* or *circle*.

- Train Model using Reinforcement Learning Framework (*i.e the Model only gets rewarded after correctly predicting the shape of a contour or image*).

- Using `cv2.Laplacian(img, cv2.CV_64F)` to get rid of wobbly lines. –Caveat: `cv2` errors out because of type mis-match.

## Credits

- Tony Beltramelli
