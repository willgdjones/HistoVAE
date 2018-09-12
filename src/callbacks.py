from keras.callbacks import Callback
import tensorflow as tf
import numpy as np


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    tensor = np.squeeze(tensor)
    height, width, channel = tensor.shape

    tensor = np.floor(tensor * 256).astype('uint8')
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(
        height=height, width=width, colorspace=channel,
        encoded_image_string=image_string
    )


class TensorBoardImage(Callback):
    def __init__(self, patches_data, tag):
        super().__init__()
        self.patches_data = patches_data
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        # Load image
        patch = self.patches_data[0]
        s = patch.shape[0]
        decoded_patch = self.model.predict(
            np.reshape(patch, (1, s, s, 3))
        )
        double_patch = np.zeros((s, 2 * s, 3))
        double_patch[:, 0:s, :] = 255 - patch
        double_patch[:, s:, :] = 255 - decoded_patch

        image = make_image(double_patch)
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=self.tag, image=image)]
        )
        writer = tf.summary.FileWriter('./tensorboardlogs')
        writer.add_summary(summary, epoch)
        writer.close()
