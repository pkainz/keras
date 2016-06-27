from keras import backend as K
from keras.backend.theano_backend import _on_gpu
from keras.layers.core import Dense, Layer
from keras.engine.topology import InputSpec
from keras.layers.convolutional import UpSampling2D, UpSampling3D, Convolution2D, Convolution3D
from theano import tensor as T
from theano.tensor.nnet import conv3d2d
from theano.sandbox.cuda import dnn


class SumLayer(Layer):
    def __init__(self, **kwargs):
        super(SumLayer, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1, input_shape[2], input_shape[3])

    def call(self, x, mask=None):
        return K.sum(x, axis=1, keepdims=True)


class DePool2D(UpSampling2D):
    '''Simplar to UpSample, yet traverse only pooled elements

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, channels, upsampled_rows, upsampled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, upsampled_rows, upsampled_cols, channels)` if dim_ordering='tf'.

    # Arguments
        size: tuple of 2 integers. The upsampling factors for rows and columns.
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 4

    def __init__(self, master_layer, *args, **kwargs):
        self._master_layer = master_layer
        super(DePool2D, self).__init__(*args, **kwargs)

    def call(self, x, mask=None):
        output = K.resize_images(x, self.size[0], self.size[1], 
                                 self.dim_ordering)
  
        f = K.gradients(K.sum(self._master_layer.output), 
                        self._master_layer.input) * output
  
        return f

#     # old API (keras 0.3)
#     def get_output(self, train=False):
#         X = self.get_input(train)
#         output = K.resize_images(X, self.size[0], self.size[1], 
#                                  self.dim_ordering)
#   
#         f = K.gradients(K.sum(self._master_layer.get_output(train)), 
#                         self._master_layer.get_input(train)) * output
#   
#         return f


class DePool3D(UpSampling3D):
    '''Repeat the first, second and third dimension of the data
    by size[0], size[1] and size[2] respectively.

    Note: this layer will only work with Theano for the time being.

    # Arguments
        size: tuple of 3 integers. The upsampling factors for dim1, dim2 and dim3.
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 4.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".

    # Input shape
        5D tensor with shape:
        `(samples, channels, dim1, dim2, dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, dim1, dim2, dim3, channels)` if dim_ordering='tf'.

    # Output shape
        5D tensor with shape:
        `(samples, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)` if dim_ordering='tf'.
    '''
    input_ndim = 5

    def __init__(self, master_layer, *args, **kwargs):
        self._master_layer = master_layer
        super(DePool3D, self).__init__(*args, **kwargs)

    def call(self, x, mask=None):
        output = K.resize_volumes(x, self.size[0], self.size[1], self.size[2],
                                self.dim_ordering)
  
        f = K.gradients(K.sum(self._master_layer.output), 
                        self._master_layer.input) * output
        
        return f
        
#     # old API (keras 0.3)
#     def get_output(self, train=False):
#         X = self.get_input(train)
#         output = K.resize_volumes(X, self.size[0], self.size[1], self.size[2],
#                                 self.dim_ordering)
#   
#         f = K.gradients(K.sum(self._master_layer.get_output(train)), 
#                         self._master_layer.get_input(train)) * output
#         
#         return f
    


def deconv2d_fast(x, kernel, strides=(1, 1), 
                  border_mode='valid', dim_ordering='th',
                  image_shape=None, filter_shape=None):
    '''
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        x = x.dimshuffle((0, 3, 1, 2))
        kernel = kernel.dimshuffle((3, 2, 0, 1))
        if image_shape:
            image_shape = (image_shape[0], image_shape[3],
                           image_shape[1], image_shape[2])
        if filter_shape:
            filter_shape = (filter_shape[3], filter_shape[2],
                            filter_shape[0], filter_shape[1])

    if _on_gpu() and dnn.dnn_available():
        if border_mode == 'same':
            assert (strides == (1, 1))
            conv_out = dnn.dnn_conv(img=x,
                                    conv_mode='cross', # <<<<< Retain the filters, don't flip
                                    kerns=kernel,
                                    border_mode='full')
            shift_x = (kernel.shape[2] - 1) // 2
            shift_y = (kernel.shape[3] - 1) // 2
            conv_out = conv_out[:, :,
                       shift_x:x.shape[2] + shift_x,
                       shift_y:x.shape[3] + shift_y]
        else:
            conv_out = dnn.dnn_conv(img=x,
                                    conv_mode='cross', # <<<<< Retain the filters, don't flip
                                    kerns=kernel,
                                    border_mode=border_mode,
                                    subsample=strides)
    else:
        if border_mode == 'same':
            th_border_mode = 'full'
            assert (strides == (1, 1))
        elif border_mode == 'valid':
            th_border_mode = 'valid'
        else:
            raise Exception('Border mode not supported: ' + str(border_mode))

        conv_out = T.nnet.conv2d(x, kernel,
                                 border_mode=th_border_mode,
                                 subsample=strides,
                                 filter_flip=False,  # <<<<< IMPORTANT !!! don't flip kernel for tied weights (=cross correlation)
                                 input_shape=image_shape,
                                 filter_shape=filter_shape)
        if border_mode == 'same':
            shift_x = (kernel.shape[2] - 1) // 2
            shift_y = (kernel.shape[3] - 1) // 2
            conv_out = conv_out[:, :,
                       shift_x:x.shape[2] + shift_x,
                       shift_y:x.shape[3] + shift_y]
    if dim_ordering == 'tf':
        conv_out = conv_out.dimshuffle((0, 2, 3, 1))
    return conv_out


def deconv3d(x, kernel, strides=(1, 1, 1),
           border_mode='valid', dim_ordering='th',
           volume_shape=None, filter_shape=None):
    '''
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    if border_mode not in {'same', 'valid'}:
        raise Exception('Invalid border mode: ' + str(border_mode))

    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        # TF input shape: (samples, conv_dim1, conv_dim2, conv_dim3, input_depth)
        # TH kernel shape: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3)
        # TF kernel shape: (kernel_dim1, kernel_dim2, kernel_dim3, input_depth, out_depth)
        x = x.dimshuffle((0, 4, 1, 2, 3))
        kernel = kernel.dimshuffle((4, 3, 0, 1, 2))
        if volume_shape:
            volume_shape = (volume_shape[0], volume_shape[4],
                            volume_shape[1], volume_shape[2], volume_shape[3])
        if filter_shape:
            filter_shape = (filter_shape[4], filter_shape[3],
                            filter_shape[0], filter_shape[1], filter_shape[2])

    if border_mode == 'same':
        assert(strides == (1, 1, 1))
        pad_dim1 = (kernel.shape[2] - 1)
        pad_dim2 = (kernel.shape[3] - 1)
        pad_dim3 = (kernel.shape[4] - 1)
        output_shape = (x.shape[0], x.shape[1],
                        x.shape[2] + pad_dim1,
                        x.shape[3] + pad_dim2,
                        x.shape[4] + pad_dim3)
        output = T.zeros(output_shape)
        indices = (slice(None), slice(None),
                   slice(pad_dim1 // 2, x.shape[2] + pad_dim1 // 2),
                   slice(pad_dim2 // 2, x.shape[3] + pad_dim2 // 2),
                   slice(pad_dim3 // 2, x.shape[4] + pad_dim3 // 2))
        x = T.set_subtensor(output[indices], x)
        border_mode = 'valid'

    border_mode_3d = (border_mode, border_mode, border_mode)
    
    #### TRANSPOSED KERNELS ####
    # flip the filters again, since the original Convolution3D implemented in 
    # the keras backend (theano.tensor.nnet.conv3d2d.conv3d) does it as well
    # this way we emulate the transposed convolution
    ###
    # TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
    # TH kernel shape: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3)
    # idx 2, 3, 4 are kernel size dimensions
    filters_flip = kernel[:,:,::-1,::-1,::-1]
    ####
    # perform the convolution
    conv_out = conv3d2d.conv3d(signals=x.dimshuffle(0, 2, 1, 3, 4),
                               filters=filters_flip.dimshuffle(0, 2, 1, 3, 4),
                               border_mode=border_mode_3d)
    # re-arrange the dimensions of the output
    conv_out = conv_out.dimshuffle(0, 2, 1, 3, 4)

    # support strides by manually slicing the output
    if strides != (1, 1, 1):
        conv_out = conv_out[:, :, ::strides[0], ::strides[1], ::strides[2]]

    if dim_ordering == 'tf':
        conv_out = conv_out.dimshuffle((0, 2, 3, 4, 1))

    return conv_out


class Deconvolution2D(Convolution2D):
    '''Convolution operator for filtering windows of two-dimensional inputs.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

    # Examples

    ```python
        # apply a 3x3 convolution with 64 output filters on a 256x256 image:
        model = Sequential()
        model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
        # now model.output_shape == (None, 64, 256, 256)

        # add a 3x3 convolution on top, with 32 output filters:
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        # now model.output_shape == (None, 32, 256, 256)
    ```

    # Arguments
        master_layer: The layer to tie the weights to.
        nb_out_channels: Number of output channels (=input to the dependent layer).
        layer_history_index: For use with functional API.
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid' or 'same'.
        subsample: tuple of length 2. Factor by which to subsample output.
            Also called strides elsewhere.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 4

    def __init__(self, master_layer, nb_out_channels=None, 
                 layer_history_index=0, *args, **kwargs):
        """
        @param layer_history_index 
            For use with functional API (keras >=1.0), the index specifies input to the
            dependent layer should be the output of this one
        """
        try:
            # compatibility for the functional API
            self._master_layer = master_layer._keras_history[layer_history_index]
        except AttributeError as e:
            # for use with Sequential().add(...)
            self._master_layer = master_layer
        
        kwargs['nb_filter'] = self._master_layer.nb_filter
        kwargs['nb_row'] = self._master_layer.nb_row
        kwargs['nb_col'] = self._master_layer.nb_col
        super(Deconvolution2D, self).__init__(*args, **kwargs)
        
        # autocompute the output channels
        if nb_out_channels == None:
            if self.dim_ordering == 'th':
                self.nb_out_channels = self._master_layer.input_shape[1]
            elif self.dim_ordering == 'tf':
                raise NotImplementedError()
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        else:
            self.nb_out_channels = nb_out_channels

    def build(self, input_shape):
        assert len(input_shape) == 4
        
        if self.dim_ordering == 'th':
            self.W = self._master_layer.W.dimshuffle((1, 0, 2, 3))
            self.W_shape = (self.nb_out_channels, self.nb_filter, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            raise NotImplementedError()
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        if self.bias:
            self.b = K.zeros((self.nb_out_channels,))
            self.trainable_weights = [self.b]
        else:
            self.trainable_weights = []

        # if setting params here, bias is not in trainable weights!
#         self.b = K.zeros((self.nb_out_channels,))
#         self.params = [self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint


    def get_output_shape_for(self, input_shape):
        output_shape = list(super(Deconvolution2D, self).get_output_shape_for(input_shape))

        if self.dim_ordering == 'th':
            output_shape[1] = self.nb_out_channels
        elif self.dim_ordering == 'tf':
            output_shape[0] = self.nb_out_channels
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        return tuple(output_shape)


    def call(self, x, mask=None):
        conv_out = deconv2d_fast(x, self.W,
                                 strides=self.subsample,
                                 border_mode=self.border_mode,
                                 dim_ordering=self.dim_ordering,
                                 filter_shape=self.W_shape)
    
        if self.dim_ordering == 'th':
            output = conv_out + K.reshape(self.b, (1, self.nb_out_channels, 1, 1))
        elif self.dim_ordering == 'tf':
            output = conv_out + K.reshape(self.b, (1, 1, 1, self.nb_out_channels))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
    
        output = self.activation(output)
        return output
    

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(Deconvolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Deconvolution3D(Convolution3D):
    '''Convolution operator for filtering windows of three-dimensional inputs.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 10, 128, 128)` for 10 frames of 128x128 RGB pictures.

    Note: this layer will only work with Theano for the time being.

    # Arguments
        master_layer: The layer to tie the weights to.
        nb_out_channels: Number of output channels (=input to the dependent layer).
        layer_history_index: For use with functional API.
        nb_filter: Number of convolution filters to use.
        kernel_dim1: Length of the first dimension in the convolution kernel.
        kernel_dim2: Length of the second dimension in the convolution kernel.
        kernel_dim3: Length of the third dimension in the convolution kernel.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
        border_mode: 'valid' or 'same'.
        subsample: tuple of length 3. Factor by which to subsample output.
            Also called strides elsewhere.
            Note: 'subsample' is implemented by slicing the output of conv3d with strides=(1,1,1).
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 4.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
        bias: whether to include a bias (i.e. make the layer affine rather than linear).

    # Input shape
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if dim_ordering='tf'.

    # Output shape
        5D tensor with shape:
        `(samples, nb_filter, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, nb_filter)` if dim_ordering='tf'.
        `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have changed due to padding.
    '''
    input_ndim = 5
    
    def __init__(self, master_layer, nb_out_channels=None,
                 layer_history_index=0, *args, **kwargs):
        """
        @param layer_history_index 
            For use with functional API (keras >=1.0), the index specifies input to the
            dependent layer should be the output of this one
        """
        try:
            # compatibility for the functional API
            self._master_layer = master_layer._keras_history[layer_history_index]
        except AttributeError as e:
            # for use with Sequential().add(...)
            self._master_layer = master_layer
        
        kwargs['nb_filter'] = self._master_layer.nb_filter
        kwargs['kernel_dim1'] = self._master_layer.kernel_dim1
        kwargs['kernel_dim2'] = self._master_layer.kernel_dim2
        kwargs['kernel_dim3'] = self._master_layer.kernel_dim3
        super(Deconvolution3D, self).__init__(*args, **kwargs)
        
        # autocompute the output channels
        if nb_out_channels == None:
            if self.dim_ordering == 'th':
                self.nb_out_channels = self._master_layer.input_shape[1]
            elif self.dim_ordering == 'tf':
                raise NotImplementedError()
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        else:
            self.nb_out_channels = nb_out_channels


    def build(self, input_shape):
        assert len(input_shape) == 5
        self.input_spec = [InputSpec(shape=input_shape)]

        if self.dim_ordering == 'th':
            self.W = self._master_layer.W.dimshuffle((1, 0, 2, 3, 4))
            self.W_shape = (self.nb_out_channels, self.nb_filter, 
                            self.kernel_dim1, self.kernel_dim2, self.kernel_dim3)
        elif self.dim_ordering == 'tf':
            raise NotImplementedError()
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        if self.bias:
            self.b = K.zeros((self.nb_out_channels,))
            self.trainable_weights = [self.b]
        else:
            self.trainable_weights = []

        # if setting params here, bias is not in trainable weights!
#         self.b = K.zeros((self.nb_out_channels,))
#         self.params = [self.b]
        self.regularizers = []

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
            

    def get_output_shape_for(self, input_shape):
        output_shape = list(super(Deconvolution3D, self).get_output_shape_for(input_shape))

        if self.dim_ordering == 'th':
            output_shape[1] = self.nb_out_channels
        elif self.dim_ordering == 'tf':
            output_shape[0] = self.nb_out_channels
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        return tuple(output_shape)


    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        # transpose the filters, because 3D conv used by keras does not
        output = deconv3d(x, self.W, 
                          strides=self.subsample,
                          border_mode=self.border_mode,
                          dim_ordering=self.dim_ordering,
                          volume_shape=input_shape,
                          filter_shape=self.W_shape)        
        
        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, self.nb_out_channels, 1, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, 1, self.nb_out_channels))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = self.activation(output)
        return output
    
    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'kernel_dim1': self.kernel_dim1,
                  'kernel_dim2': self.kernel_dim2,
                  'kernel_dim3': self.kernel_dim3,
                  'dim_ordering': self.dim_ordering,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias}
        base_config = super(Deconvolution3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DependentDense(Dense):
    def __init__(self, output_dim, master_layer, layer_history_index=0,
                 init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, **kwargs):
        """
        @param layer_history_index 
            For use with functional API (keras >=1.0), the index specifies input to the
            dependent layer should be the output of this one
        """
        try:
            # compatibility for the functional API
            self._master_layer = master_layer._keras_history[layer_history_index]
        except AttributeError as ae:
            # for use with Sequential().add(...)
            self._master_layer = master_layer
        
        super(DependentDense, self).__init__(output_dim, **kwargs)

    def build(self, input_dim):
        self.W = self._master_layer.W.T
        self.b = K.zeros((self.output_dim,))
        self.params = [self.b]
        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

#     @classmethod
#     def from_config(cls, config):
#         config2 = super(DependentDense).from_config(cls, config)
#         return config2


