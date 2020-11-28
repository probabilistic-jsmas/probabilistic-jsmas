import tensorflow as tf

from cleverhans.picklable_model import Layer


class MaxPooling2D(Layer):
    def __init__(self, pool_size, strides, padding, **kwargs):
        super().__init__(**kwargs)

        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

        self.input_shape = None
        self.output_shape = None

    def set_input_shape(self, shape):
        self.input_shape = shape

        self.output_shape = (
            shape[0],
            int((shape[1]-self.pool_size[0])/self.strides[0]) + 1,
            int((shape[2]-self.pool_size[1])/self.strides[0]) + 1,
            shape[-1]
        )

    def get_params(self):
        return []

    def fprop(self, x, **kwargs):
        strides = (1,) + self.strides + (1,)
        pool_size = (1,) + self.pool_size + (1,)

        return tf.nn.max_pool2d(x, pool_size, strides, padding=self.padding, **kwargs)
