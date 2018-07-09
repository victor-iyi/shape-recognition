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
import os
from detector.base import Base
import numpy as np
import keras

import utils


class Network(Base):
    """A six-layer Neural Network for detecting how many shapes is
    contained in an image.

    Methods
    -------
        predict(filenames, **kwargs):
            Predicting shape numbers given file names.

        train(X, y, **kwargs):
            Train model on labelled data.

            Keyword Arguments:
                epochs (int): Number of training epochs. default=10
                batch_size: Mini-batch size. default=128
                optimizer (str, keras.optimizers.Optimzier): Network
                    optimizer. default='rmsprop'
                valid_portion (float): Validation size (in fraction).
                    between 0 & 1. default=0.
                save_best_only (bool): Save only the best recorded accuracy
                    during training. default=True
                steps_per_epoch (int): Number of iteration per epoch.
                    default=10,000.

    Examples
    --------
        ```python
        >>> model = Network()
        ```

    Attributes
    ----------
        model (keras.Model) - A keras Sequential model object.

    """

    def __init__(self, **kwargs):
        """Network initialization. Create new Network object.

        Keyword Arguments:
            input_shape (tuple): Input image shape.
                (default=(200, 200, 5))
            dropout (float): Rate of turning off neurons randomly,
                during training. (default=0.5)
            verbose (int): Logging verbosity values between 0 & 1.
                (default=0)
        """
        super(Network, self).__init__(**kwargs)

        input_shape = kwargs.get('input_shape', (200, 200, 3))
        dropout = kwargs.get('dropout', 0.5)

        # Sequential model.
        self._model = keras.models.Sequential()

        # Convolutional layers (feature extraction).
        self._model.add(keras.layers.Conv2D(filters=16, kernel_size=5,
                                            input_shape=input_shape,
                                            strides=2, activation='relu'))
        self._model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        self._model.add(keras.layers.Conv2D(filters=64, kernel_size=5,
                                            strides=2, activation='relu'))
        self._model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        self._model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                                            strides=2, activation='relu'))

        # Dropout & Flatten convolutional layers.
        self._model.add(keras.layers.Dropout(rate=dropout))
        self._model.add(keras.layers.Flatten())

        # Fully connected layers.
        self._model.add(keras.layers.Dense(128, activation='elu'))
        self._model.add(keras.layers.Dense(32, activation='elu'))

        # Shape prediction layer.
        self._model.add(keras.layers.Dense(3))

        self._model.summary()

    def __repr__(self):
        return 'Network()'

    def load(self, filepath, **kwargs):
        self._model = keras.models.load_model(filepath, **kwargs)

    def _predict(self, X, **kwargs):
        """Make predictions given image file paths.

        Arguments:
            filenames {tuple} -- Iterable list of file paths.

        Returns:
            {np.ndarray} -- array-like contianing predicted values.
        """
        return self._model.predict(X, **kwargs)

    def train(self, X, y, **kwargs):
        """Train model on labelled data.

        Arguments:
            X {list} -- List of path to training images.
            y {list} -- List of paths to training labels.

        Keyword Arguments:
            epochs (int): Number of training epochs. default=5
            batch_size: Mini-batch size. default=128
            optimizer (str, keras.optimizers.Optimzier): Network
                optimizer. default='rmsprop'
            valid_portion (float): Validation size (in fraction).
                between 0 & 1. default=0.
            save_best_only (bool): Save only the best recorded accuracy
                during training. default=True
            steps_per_epoch (int): Number of iteration per epoch.
                default=1,000.
        """
        # Extract keyword arguments.
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 128)
        optimizer = kwargs.get('optimizer', 'rmsprop')
        save_dir = kwargs.get('save_dir', 'saved/models')
        valid_portion = kwargs.get('valid_portion', 0.)
        save_best_only = kwargs.get('save_best_only', True)
        steps_per_epoch = kwargs.get('steps_per_epoch', 1000)

        # Make sure valid_portion is between 0 & 1.
        assert 0 <= valid_portion < 1, '`valid_portion` must be between 0. & 1.'

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # Split data into training & validation set.
        size = int(valid_portion * len(X))
        if size > 0:
            X_train, y_train = X[:-size], y[:-size]
            X_valid, y_train = X[-size:], y[-size:]

            # Create data generator for train & validation set.
            train_gen = utils.DataGenerator(X[:-size], y[:-size], batch_size)
            val_gen = utils.DataGenerator(X[-size:], y[-size:], batch_size)
        else:
            # Create data generator for only training set.
            train_gen = utils.DataGenerator(X, y, batch_size)
            val_gen = None

        # Checkpoint callback.
        checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(save_dir, 'model.{epoch:03d}.h5'),
                                                     save_best_only=save_best_only,
                                                     verbose=self._verbose)

        # Compile model using Adam optimizer & cross entropy loss function.
        self._model.compile(loss='categorical_crossentropy',
                            optimizer=optimizer)

        # Train model.
        try:
            self._model.fit_generator(generator=train_gen, epochs=epochs,
                                      validation_data=val_gen,
                                      steps_per_epoch=steps_per_epoch,
                                      callbacks=[checkpoint], verbose=self._verbose)
        except KeyboardInterrupt:
            print('\n{}'.format('-' * 65))
            print('Interrupted by user! \nSaving model...')

            # Save model.
            keras.models.save_model(model=self._model,
                                    filepath=os.path.join(save_dir, 'model.interrupt.h5'))

            print('{}\n'.format('-' * 55))

    @property
    def model(self):
        return self._model
