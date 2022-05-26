import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

def conv_block(inputs, num_filters):
    """Creates a convoltuional block.

    Args:
        inputs: the data for the layer
        num_filters: the filter size for the convolution
    
    Returns:
        a convolution block
    """
    x = Conv2D(num_filters, 3, padding ="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding ="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    """Creates an encoding block

    Args:
        inputs: the data for the layer
        num_filters: the filter size for the convolution
    
    Returns:
        An encoding block
    """
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2,2))(x)

    return x, p

def decoder_block(inputs, skip_features, num_filters):
    """Creates an decoding block

    Args:
        inputs: the data for the layer
        num_filters: the filter size for the convolution
    
    Returns:
        An encoding block
    """
    x = Conv2DTranspose(num_filters, (2,2), strides=(2,2), padding="same")(inputs)
    x = Concatenate(axis=3)([x, skip_features])
    x = conv_block(x, num_filters)

    return x

def build_unet(input_shape, num_channels=4):
    """Initializes a Simple Unit

    Args:
        input_shape: the size of the image
        num_channels: the number of output channels

    Returns:
        a built U-Net model 
    """

    inputs = Input(input_shape)

    """ Encoder """
    s1, p1 = encoder_block(inputs, 16)
    s2, p2 = encoder_block(p1, 32)

    """ Bridge """

    b1 = conv_block(p2, 64)

    """ Decoder """
    d1 = decoder_block(b1, s2, 32)
    d2 = decoder_block(d1, s1, 16)


    """ Output """

    outputs = Conv2D(num_channels, (1,1), padding="same")(d2)

    model = Model(inputs, outputs, name="U-Net")

    return model
