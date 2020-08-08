import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from functools import partial
from .layers import *
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import UpSampling2D, concatenate, Input, Activation, Add, Conv2D, Lambda


class Generator:
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
                filters=64,
                kernel_size=9,
                strides=1,
                activation='prelu',
            )

            x_ = tf.identity(pre_blocks)

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
            GRL = tf.add(post_blocks, x_)
            y = conv_2d(
                x=GRL,
                filters=3,
                kernel_size=9,
                strides=1,
                activation='tanh'
            )
            return y

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

    def _RRDB(self, input_layer, t):
        x = input_layer

        for d in range(1, self.D + 1):
            LFF = self._dense_block(x, d, t)
            LFF_beta = Lambda(lambda x: x * self.beta)(LFF)
            x = tf.add(x, LFF_beta)
        x = Lambda(lambda x: x * self.beta)(x)
        x = tf.add(input_layer, x)
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
                filters=self.G,
                kernel_size=self.kernel_size,
                strides=1,
                activation='relu',
            )

            # x = tf.concat([x, F_dc], 3)

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



class Discriminator:
	def __init__(self, patch_size, kernel_size=3, name='rdn_discriminator'):
		self.patch_size = patch_size
		self.kernel_size = kernel_size
		self.block_param = {}
		self.block_param['filters'] = (64, 128, 128, 256, 256, 512, 512)
		self.block_param['strides'] = (2, 1, 2, 1, 2, 1, 2)
		self.block_num = len(self.block_param['filters'])
		self.name = name
		self.training = True

	def __call__(self, x, reuse=True):

		with tf.variable_scope(self.name) as vs:
			if reuse:
				vs.reuse_variables()

			print(x.shape)
			x = conv_block_2d(
				x, 
				filters=64, 
				strides=1,
				batch_norm=False,
				activation='leaky_relu',
				kernel_size=self.kernel_size,
				training=self.training
			)

			for i in range(self.block_num):
				x = conv_2d(
					x,
					filters=self.block_param['filters'][i],
					strides=self.block_param['strides'][i],
					activation='leaky_relu',
					kernel_size=self.kernel_size
				)

			x = flatten(x)
			x = dense(
				x=x,
				units=self.block_param['filters'][i]*2,
				activation='leaky_relu'
			)
			x = dense(
				x=x,
				units=1,
				activation='sigmoid'
			)
			return x


	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]