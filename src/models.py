from keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D,
    UpSampling2D, Flatten, Reshape, Dropout
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
    def __init__(self, inner_dim, dropout_rate=0):
        self.inner_dim = inner_dim
        self.dropout_rate = dropout_rate

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
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)

        encoded = Dense(
            self.inner_dim,
            activity_regularizer='l2'
        )(x)
        return encoded

    def decode(self, input):
        x = input
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        x = Dense(self.middle_dim)(x)
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
        self.params = params
        self.params['N'] = N

        self.name = (
            f"CA_ps{self.params['patch_size']}_"
            f"n{self.params['N']}_e{self.params['epochs']}_"
            f"lr{self.params['lr']}_bs{self.params['batch_size']}_"
            f"dim{self.params['inner_dim']}_do{self.params['dropout_rate']}"
        )
        input_img = Input(
            shape=(
                self.params['patch_size'],
                self.params['patch_size'],
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
                        f'./tensorboardlogs/{self.name}'
                    )
                ),
            ],
        )
        self.model = model

    def save(self):
        assert self.model, "Model must be trained first"
        self.model.save("models/" + self.name + ".pkl")
