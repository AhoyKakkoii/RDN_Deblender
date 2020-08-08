import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from functools import partial
from .layers import *
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import UpSampling2D, concatenate, Input, Activation, Add, Conv2D, Lambda


class BaseModel:
    """
    Abstract neural network model class.
    """
    def __call__(self, x, params, reuse=False):
        """
        Forward pass through network.

        Args:
            x(tf.Tensor): input tensor.
            name(str): name for variable scope definition.
        """
        raise NotImplementedError("Abstract class methods should not be called.")

    @property
    def vars(self):
        """
        Getter function for model variables.
        """
        raise NotImplementedError("Abstract class methods should not be called.")


conv_block = partial(
    conv_block_2d,
    kernel_size=3,
    activation='leaky_relu'
)


class Discriminator(BaseModel):
    def __init__(self, name='discriminator'):
        self.name = name
        self.training = True

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            x = conv_2d(
                x=x,
                filters=64,
                kernel_size=3,
                strides=1,
                activation='leaky_relu'
            )

            x = conv_block(x=x, filters=64, strides=2, training=self.training)
            x = conv_block(x=x, filters=128, strides=1, training=self.training)
            x = conv_block(x=x, filters=128, strides=2, training=self.training)
            x = conv_block(x=x, filters=256, strides=1, training=self.training)
            x = conv_block(x=x, filters=256, strides=2, training=self.training)
            #x = conv_block(x=x, filters=512, strides=1, training=self.training)
            #x = conv_block(x=x, filters=512, strides=2, training=self.training)
            #x = conv_block(x=x, filters=1024, strides=1, training=self.training)
            #x = conv_block(x=x, filters=1024, strides=2, training=self.training)


            x = flatten(x)

            x = dense(
                x=x,
                units=2048,
                activation='leaky_relu'
                )
            x = dense(
                x=x,
                units=1,
                activation='linear'
                )

            return x

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(BaseModel):
    def __init__(self, num_root_blocks=10, num_branch_blocks=6,
        name='generator'):

        self.num_root_blocks = num_root_blocks
        self.num_branch_blocks = num_branch_blocks
        self.name = name
        self.training = True

    def __call__(self, x):
        with tf.variable_scope(self.name) as vs:

            """
            Root
            """

            x = conv_2d(
                x=x,
                filters=64,
                kernel_size=9,
                strides=1,
                activation='prelu',
            )

            x_ = tf.identity(x)

            for i in range(self.num_root_blocks):

                x = res_block_2d(
                    x=x,
                    kernel_size=3,
                    activation='prelu',
                    training=self.training,
                )

            x = conv_2d(
                x=x,
                filters=64,
                kernel_size=3,
                strides=1,
                activation='linear',
            )

            x = batch_norm(x, training=self.training)

            x_split = tf.add(x, x_)

            """
            Branch One
            """

            y = res_block_2d(
                x=x_split,
                kernel_size=3,
                activation='prelu',
                training=self.training
            )

            for i in range(self.num_branch_blocks - 1):
                y = res_block_2d(
                    x=y,
                    kernel_size=3,
                    activation='prelu',
                    training=self.training
                )

            y = conv_2d(
                x=y,
                filters=3,
                kernel_size=9,
                strides=1,
                activation='tanh'
            )

            """
            Branch Two
            """
            '''
            z = res_block_2d(
                x=x_split,
                kernel_size=3,
                activation='prelu',
                training=self.training
            )

            for i in range(self.num_branch_blocks - 1):
                z = res_block_2d(
                    x=z,
                    kernel_size=3,
                    activation='prelu',
                    training=self.training
                )

            z = conv_2d(
                x=z,
                filters=3,
                kernel_size=9,
                strides=1,
                activation='tanh'
            )

            return y, z
            '''
            return y

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class VGG19:
    def __init__(self, name='vgg'):
        self.name = name

    def initialize(self):
        with tf.variable_scope(self.name) as vs:
            model = k.applications.VGG19()
            self.vgg19 = k.models.Model(
                inputs=model.input,
                outputs=model.layers[20].output
            )

    def __call__(self, x):
        with tf.variable_scope(self.name) as vs:
            x = tf.image.resize_images(
                x,
                size=[224, 224],
                method=0,
                align_corners=False,
            )

            x = (x + 1)/2.
            x = 255. * x

            VGG_MEAN = [103.939, 116.779, 123.68]

            if tf.__version__ <= '0.11':
                red, green, blue = tf.split(3, 3, x)
            else:
                red, green, blue = tf.split(x, 3, 3)

            assert red.get_shape().as_list()[1:] == [224, 224, 1]
            assert green.get_shape().as_list()[1:] == [224, 224, 1]
            assert blue.get_shape().as_list()[1:] == [224, 224, 1]

            if tf.__version__ <= '0.11':
                bgr = tf.concat(3, [
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    red - VGG_MEAN[2],
                ])
            else:
                bgr = tf.concat(
                    [
                        blue - VGG_MEAN[0],
                        green - VGG_MEAN[1],
                        red - VGG_MEAN[2],
                    ], axis=3)

            assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

            return self.vgg19(bgr)

class RRDN:
    def __init__(
        self, 
        params={}, 
        patch_size=None, 
        beta=0.2, 
        c_dim=3, 
        kernel_size=3, 
        init_val=0.05,
        name='rrdn'
    ):
        self.params = params
        self.beta = beta
        self.c_dim = c_dim
        self.C = self.params['C']
        self.D = self.params['D']
        self.G = self.params['G']
        self.G0 = self.params['G0']
        self.T = self.params['T']
        self.initializer = RandomUniform(
            minval=-init_val, maxval=init_val, seed=None)
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.name = name

    def __call__(self, x):
        with tf.variable_scope(self.name) as vs:

            pre_blocks = conv_2d(
                x=x,
                filters=self.G0,
                kernel_size=self.kernel_size,
                strides=1,
                activation='prelu',
            )

            for t in range(1, self.T + 1):
                if t == 1:
                    x = self._RRDB(pre_blocks, t)
                else:
                    x = self._RRDB(x, t)

            post_blocks = conv_2d(
                x=x,
                filters=self.G0,
                kernel_size=self.kernel_size,
                strides=1,
                activation='prelu',
            )
            # Global Residual Learning
            GRL = Add(name='GRL')([post_blocks, pre_blocks])
            # Upscaling
            PS = self._pixel_shuffle(GRL)
            # Compose SR image
            SR = conv_2d(
                x=PS,
                filters=self.G0,
                kernel_size=self.kernel_size,
                strides=1,
                activation='prelu',
            )

            return SR

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

    def _RRDB(self, input_layer, t):
        x = input_layer

        for d in range(1, self.D + 1):
            LFF = self._dense_block(x, d, t)
            LFF_beta = Lambda(lambda x: x * self.beta)(LFF)
            x = Add(name='LRL_%d_%d' % (t, d))([x, LFF_beta])
        x = Lambda(lambda x: x * self.beta)(x)
        x = Add(name='RRDB_%d_out' % (t))([input_layer, x])
        return x

    def _dense_block(self, input_layer, d, t):
        """
        Implementation of the (Residual) Dense Block as in the paper
        Residual Dense Network for Image Super-Resolution (Zhang et al. 2018).

        Residuals are incorporated in the RRDB.
        d is an integer only used for naming. (d-th block)
        """

        x = input_layer
        for c in range(1, self.C + 1):
            F_dc = conv_2d(
                x=x,
                filters=self.G0,
                kernel_size=self.kernel_size,
                strides=1,
                activation='relu',
            )

            x = concatenate([x, F_dc], axis=3)

        # DIFFERENCE: in RDN a kernel size of 1 instead of 3 is used here
        x = conv_2d(
                x=x,
                filters=self.G0,
                kernel_size=self.kernel_size,
                strides=1,
                activation='relu',
            )
        return x

    def _pixel_shuffle(self, input_layer):
        """ PixelShuffle implementation of the upscaling part. """

        x = conv_2d(
                x=input_layer,
                filters=self.G0,
                kernel_size=self.kernel_size,
                strides=1,
                activation='relu',
            )
        return Lambda(
            lambda x: tf.nn.depth_to_space(
                x, block_size=2, data_format='NHWC'),
            name='PixelShuffle',
        )(x)
