from keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np


class ConvolutionalAutoencoder():
    def __init__(self, dim, patchsize):
        self.dim = dim
        self.patchsize = patchsize

    def encode(self, input_img):
        x = Conv2D(
            32, (3, 3), activation='relu',
            padding='same'
        )(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(
            32, (3, 3), activation='relu',
            padding='same'
        )(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        self.middle_tensor_shape = x.shape.as_list()[1:]
        self.middle_dim = np.prod(self.middle_tensor_shape)
        x = Flatten()(x)

        encoded = Dense(self.dim, activity_regularizer='l2')(x)
        return encoded

    def decode(self, input):
        # input = Input(shape=(self.dim,))
        x = Dense(self.middle_dim)(input)
        x = Reshape(self.middle_tensor_shape)(x)
        x = Conv2D(
            32, (3, 3), activation='relu',
            padding='same'
        )(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(
            32, (3, 3),
            activation='relu', padding='same'
        )(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(
            3, (3, 3), activation='sigmoid',
            padding='same'
        )(x)
        return decoded

    def fit_generator(self, data):
        input_img = Input(shape=(self.patchsize, self.patchsize, 3))
        model = Model(
            input_img,
            self.decode(self.encode(input_img))
        )
        model.compile(optimizer='adadelta', loss='binary_crossentropy')
        # datagen = ImageDataGenerator(
        #     # featurewise_center=True,
        #     # featurewise_std_normalization=True,
        #     # rotation_range=20,
        #     # width_shift_range=0.2,
        #     # height_shift_range=0.2,
        #     horizontal_flip=True,
        #     vertical_flip=True
        # )
        # datagen.flow(generator, batch_size=32)
        batchsize = 50
        model.fit(
            data,
            data,
            batch_size=batchsize,
            epochs=10,
        )
