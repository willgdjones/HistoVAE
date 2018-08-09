from keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D,
    UpSampling2D, Flatten, Reshape
)
# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import (
    TensorBoard, ModelCheckpoint
)
import numpy as np
import logging
from itertools import cycle

logger = logging.getLogger(__name__)


class ConvolutionalAutoencoder():
    def __init__(self, dim, patchsize):
        self.dim = dim
        self.patchsize = patchsize
        self.name = 'ConvolutionalAutoencoder'

    def encode(self, input_img):
        x = Conv2D(
            64, (3, 3), activation='relu',
            padding='same'
        )(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(
            64, (3, 3), activation='relu',
            padding='same'
        )(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        self.middle_tensor_shape = x.shape.as_list()[1:]
        self.middle_dim = np.prod(
            self.middle_tensor_shape
        )
        x = Flatten()(x)

        encoded = Dense(
            self.dim,
            activity_regularizer='l1'
        )(x)
        return encoded

    def decode(self, input):
        x = Dense(self.middle_dim)(input)
        x = Reshape(self.middle_tensor_shape)(x)
        x = Conv2D(
            64, (3, 3), activation='relu',
            padding='same'
        )(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(
            64, (3, 3),
            activation='relu', padding='same'
        )(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(
            3, (3, 3), activation='sigmoid',
            padding='same'
        )(x)
        return decoded

    def train_on_data(self, train_data, val_data,
                      batch_size, epochs):

        input_img = Input(
            shape=(
                self.patchsize,
                self.patchsize,
                3
            )
        )
        model = Model(
            input_img,
            self.decode(self.encode(input_img))
        )
        model.compile(
            optimizer='adadelta',
            loss='binary_crossentropy',
        )

        logger.debug('Fitting model')
        model.fit(
            train_data,
            validation_data=val_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                TensorBoard(
                    log_dir=(
                        './tensorboardlogs'
                    ),
                    histogram_freq=1
                ),
            ],
            shuffle=True
        )
        self.model = model

    def train_on_generator(self, train_gen, val_gen,
                           split, N, batchsize, epochs):
        input_img = Input(
            shape=(
                self.patchsize,
                self.patchsize,
                3
            )
        )
        self.N = N
        logger.debug('Building model')
        model = Model(
            input_img,
            self.decode(self.encode(input_img))
        )
        logger.debug('Compiling model')
        model.compile(
            optimizer='adadelta',
            loss='binary_crossentropy',
        )
        # datagen = ImageDataGenerator(
        #     horizontal_flip=True,
        #     vertical_flip=True
        # )
        # datagen.flow(generator, batch_size=32)
        # epochs = 10
        # for e in range(epochs):
        #     print('Epoch', e)
        #     batches = 0
        #     for batch in datagen.flow(train_gen(), batch_size=32):
        #         import pdb; pdb.set_trace()
        #         model.fit(*batch)
        #         batches += 1
        #         if batches >= len(N) / b:
        #             # we need to break the loop by hand because
        #             # the generator loops indefinitely
        #             break
        logger.debug('Fitting model')
        model.fit_generator(
            cycle(train_gen()),
            validation_data=cycle(val_gen()),
            steps_per_epoch=(
                (1 - split) * N
            ) / batchsize,
            epochs=epochs,
            validation_steps=(
                split * N
            ) / batchsize,
            callbacks=[
                TensorBoard(
                    log_dir=(
                        './tensorboardlogs'
                    ),
                    histogram_freq=1
                ),
                ModelCheckpoint(
                    filepath=f'models/{self.name}',
                    period=5
                )
            ]
            # use_multiprocessing=True
        )
        self.model = model

    def save(self):
        assert self.model, "Model must be trained first"
        self.model.save(
            'models/CA_{self.patchsize}_{self.N},pkl'
        )
