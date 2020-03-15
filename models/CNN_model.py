from keras import utils as keras_utils
from keras import backend, engine
from tensorflow.keras import layers, models
from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/releases/download/v0.6/')


def get_input_shape(input_shape, default_size, data_format, require_flatten, weights=None):

    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            default_shape = (input_shape[0], default_size, default_size)
        else:
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)

    if weights == 'imagenet' and require_flatten:
        return default_shape

    if not input_shape:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    return input_shape


def model_DR(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet',
              input_tensor=None, pooling=None, classes=1000):

    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = get_input_shape(input_shape, default_size=default_size, require_flatten=include_top, weights=weights,
                              data_format=backend.image_data_format())

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)

    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if rows != cols or rows not in [128, 160, 192, 224]:
            if rows is None:
                rows = 224

    if backend.image_data_format() != 'channels_last':
        backend.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = conv(img_input, 32, alpha, strides=(2, 2))
    x = conv_depthwise(x, 64, alpha, depth_multiplier, block_id=1)

    x = conv_depthwise(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = conv_depthwise(x, 128, alpha, depth_multiplier, block_id=3)

    x = conv_depthwise(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = conv_depthwise(x, 256, alpha, depth_multiplier, block_id=5)

    x = conv_depthwise(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = conv_depthwise(x, 512, alpha, depth_multiplier, block_id=7)
    x = conv_depthwise(x, 512, alpha, depth_multiplier, block_id=8)
    x = conv_depthwise(x, 512, alpha, depth_multiplier, block_id=9)
    x = conv_depthwise(x, 512, alpha, depth_multiplier, block_id=10)
    x = conv_depthwise(x, 512, alpha, depth_multiplier, block_id=11)

    x = conv_depthwise(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = conv_depthwise(x, 1024, alpha, depth_multiplier, block_id=13)

    if include_top:
        if backend.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        x = GlobalAveragePooling2D()(x)
        x = layers.Reshape(shape, name='reshape_1')(x)
        x = layers.Dropout(dropout, name='dropout')(x)
        x = layers.Conv2D(classes, (1, 1),
                          padding='same',
                          name='conv_preds')(x)
        x = layers.Activation('softmax', name='act_softmax')(x)
        x = layers.Reshape((classes,), name='reshape_2')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    if input_tensor is not None:
        if hasattr(keras_utils, 'get_source_inputs'):
            get_source_inputs = keras_utils.get_source_inputs
        else:
            get_source_inputs = engine.get_source_inputs
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = models.Model(inputs, x, name='mobilenet_%0.2f_%s' % (alpha, rows))

    if weights == 'imagenet':
        if backend.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_first" format '
                             'are not available.')
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'

        if include_top:
            model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = keras_utils.get_file(model_name,
                                                weight_path,
                                                cache_subdir='models')
        else:
            model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = keras_utils.get_file(model_name,
                                                weight_path,
                                                cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    if old_data_format:
        backend.set_image_data_format(old_data_format)
    return model


def conv(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel, padding='valid', use_bias=False, strides=strides, name='conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return layers.ReLU(6., name='conv1_relu')(x)


def conv_depthwise(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id)(inputs)

    x = layers.DepthwiseConv2D((3, 3), padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)
