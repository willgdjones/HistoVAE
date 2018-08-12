from keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D,
    UpSampling2D, Flatten, Reshape
)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import (
    TensorBoard, ModelCheckpoint
)
from keras.layers import LeakyReLU
from keras.optimizers import Adam
import numpy as np
import logging
from itertools import cycle
from tqdm import tqdm
logger = logging.getLogger(__name__)


class ConvolutionalAutoencoder():
    def __init__(self, dim, patchsize):
        self.dim = dim
        self.patchsize = patchsize
        self.name = 'ConvolutionalAutoencoder'

    def encode(self, input_img):
        x = Conv2D(
            16, (3, 3),
            padding='same'
        )(input_img)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(
            32, (3, 3), padding='same'
        )(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(
            64, (3, 3), padding='same'
        )(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(
            128, (3, 3), padding='same'
        )(x)
        x = LeakyReLU()(x)
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
            128, (3, 3), padding='same'
        )(x)
        x = LeakyReLU()(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(
            64, (3, 3), padding='same'
        )(x)
        x = LeakyReLU()(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(
            32, (3, 3), padding='same'
        )(x)
        x = LeakyReLU()(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(
            16, (3, 3), padding='same'
        )(x)
        x = LeakyReLU()(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(
            3, (3, 3), activation='sigmoid',
            padding='same'
        )(x)
        return decoded

    def train_on_data(self, patches_data,
                      params, split=0.2):

        N = patches_data.shape[0]
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

        adam = Adam(
            lr=params['lr'], beta_1=params['beta_1']
        )

        model.compile(
            optimizer=adam,
            loss='mean_squared_error',
        )

        datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True
        )

        logger.debug('Fitting model')
        model.fit_generator(
            datagen.flow(
                patches_data, patches_data, batch_size=params['batch_size']
            ),
            steps_per_epoch=N / params['batch_size'], epochs=params['epochs'],
            callbacks=[
                TensorBoard(
                    log_dir=(
                        './tensorboardlogs'
                    )
                ),
            ],
        )


        # checkpoint = int(N * (1 - split))
        # train_patches = patches_data[:checkpoint, :, :, :]
        # val_patches = patches_data[checkpoint:, :, :, :]
        # train_datagen = ImageDataGenerator(
        #     horizontal_flip=True,
        #     vertical_flip=True
        # )
        # val_datagen = ImageDataGenerator()
        #
        # for e in range(epochs):
        #     print('Epoch', e)
        #     batches = 0
        #     pbar = tqdm(len(patches_data) / batch_size)
        #     for (x_train_batch, y_train_batch), (x_val_batch, y_val_batch) in\
        #         zip(
        #             train_datagen.flow(
        #                 train_patches, train_patches, batch_size=batch_size
        #             ),
        #             val_datagen.flow(
        #                 val_patches, val_patches, batch_size=batch_size)
        #             ):
        #         model.fit(
        #             train_patches, train_patches,
        #             validation_data=(val_patches, val_patches),
        #             batch_size=batch_size,
        #             callbacks=[
        #                 TensorBoard(
        #                     log_dir=(
        #                         './tensorboardlogs'
        #                     ),
        #                     histogram_freq=1
        #                 ),
        #             ],
        #         )
        #         pbar.update(1)
        #         batches += 1
        #         if batches >= len(patches_data) / batch_size:
        #             break
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
            f'models/CA_ps{self.patchsize}_n{self.N}_e{self.epochs}.pkl'
        )
