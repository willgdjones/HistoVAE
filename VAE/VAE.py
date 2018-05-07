'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.
# Reference
- Auto-Encoding Variational Bayes
  https://arxiv.org/abs/1312.6114
'''
import numpy as np
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist, cifar10
import sys
rootdir = '/hps/nobackup/research/stegle/users/willj/HistoVAE'
sys.path.append(rootdir)
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint



np.random.seed(42)

# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3
batch_size = 32


dataset = '128-patches'

if dataset == 'mnist':
    img_rows, img_cols, img_chns = 28, 28, 1
elif dataset == 'cifar10':
    img_rows, img_cols, img_chns = 32, 32, 3
elif dataset == 'histo-dev':
    img_rows, img_cols, img_chns = 256, 256, 3
elif dataset == '128-patches':
    img_rows, img_cols, img_chns = 128, 128, 3

if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)




latent_dim = 16
intermediate_dim = 512
epsilon_std = 1.0
epochs = 100
lr = 0.0001

x = Input(shape=original_img_size)
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon


# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later


decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * int(img_rows/2) * int(img_rows/2), activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, int(img_rows/2), int(img_rows/2))
else:
    output_shape = (batch_size, int(img_rows/2), int(img_rows/2), filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 29, 29)
else:
    output_shape = (batch_size, 29, 29, filters)
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)




# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x



y = CustomVariationalLayer()([x, x_decoded_mean_squash])
vae = Model(x, y)

rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
# sgd = optimizers.sgd(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
vae.compile(optimizer=rmsprop, loss=None)



if __name__ == '__main__':
    vae.summary()
    if dataset == 'mnist':
        (x_train, _), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype('float32') / 255.
        x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape((x_test.shape[0],) + original_img_size)



        print('x_train.shape:', x_train.shape)

    elif dataset == 'histo-dev':
        import h5py
        filename = '/hps/nobackup/research/stegle/users/willj/GTEx/data/h5py/patches-all_s-10_p-50.hdf5'
        with h5py.File(filename) as f:
            patches = f['/patches'].value
            labels = f['/labels'].value

        patches = [resize(x, (256, 256)) for x in patches]
        x_train, x_test, y_train, y_test = train_test_split(patches, labels, test_size=0.4, random_state=42)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

        x_train = np.array(x_train, dtype='float32')
        x_test = np.array(x_test, dtype='float32')
        x_val = np.array(x_val, dtype='float32')

        # import pdb; pdb.set_trace()

        print('x_train.shape:', x_train.shape)

    elif dataset == 'cifar10':
        (x_train, _), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

    if dataset == '128-patches':
        import h5py


        filename = '/hps/nobackup/research/stegle/users/willj/GTEx/data/patches/Lung/GTEX-144GM-0126_128.hdf5'
        with h5py.File(filename) as f:
            patches1 = f['/patches'].value
            labels1 = ['144GM'] * len(patches1)

        filename = '/hps/nobackup/research/stegle/users/willj/GTEx/data/patches/Lung/GTEX-13N1W-0726_128.hdf5'
        with h5py.File(filename) as f:
            patches2 = f['/patches'].value
            labels2 = ['13N1W'] * len(patches2)

        patches = np.concatenate([patches1, patches2])
        labels = labels1 + labels2

        x_train, x_test, y_train, y_test = train_test_split(patches, labels, test_size=0.4, random_state=42)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

        x_train = np.array(x_train, dtype='float32') / 255.
        x_test = np.array(x_test, dtype='float32') / 255.
        x_val = np.array(x_val, dtype='float32') / 255.
        import pdb; pdb.set_trace()



    vae.fit(x_train,shuffle=True,epochs=epochs,
        batch_size=batch_size,validation_data=(x_test, None),
        callbacks=[
            TensorBoard(log_dir='logs'),
            EarlyStopping(patience=10)
        ])

    vae.save('models/vae-{dataset}.hdf5'.format(dataset=dataset))



    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
    encoder.save('models/encoder-{dataset}.hdf5'.format(dataset=dataset))

    # display a 2D plot of the digit classes in the latent space
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    # plt.figure(figsize=(6, 6))
    # plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    # plt.colorbar()
    # plt.show()

    # build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _hid_decoded = decoder_hid(decoder_input)
    _up_decoded = decoder_upsample(_hid_decoded)
    _reshape_decoded = decoder_reshape(_up_decoded)
    _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
    _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
    _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
    _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
    generator = Model(decoder_input, _x_decoded_mean_squash)
    generator.save('models/generator-{dataset}.hdf5'.format(dataset=dataset))
