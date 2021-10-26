import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, Input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD,Adam

def create_base_model(classifier_embeddings,num_classes):
    input_shape = (30, 30, 3)
    img_input = Input(shape=input_shape)
    
    y = Conv2D(32, (5, 5), activation='relu', padding='same', name='block1_conv1')(img_input)
    y = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(y)
    y = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(y)
    y = Dropout(0.25)(y)

    y = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(y)
    y = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(y)
    y = Dropout(0.25)(y)

    y = Flatten(name='flatten')(y)
    y = Dense(units = classifier_embeddings,name='embedding')(y)
    y = Activation(activation='relu')(y)
    y = Dense(units = num_classes,name='logits')(y)
    y = Activation('softmax')(y)
    base_model=Model(img_input, y, name = 'shallow_cnn')
    base_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=Adam(),
                 metrics=['accuracy'])
    return base_model
