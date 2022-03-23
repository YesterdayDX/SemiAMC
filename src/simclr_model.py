import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,BatchNormalization,Activation

def attach_simclr_head(base_model, hidden_1=128, hidden_2=128, hidden_3=64):
    """
    Attach a 3-layer fully-connected encoding head

    Architecture:
        base_model
        -> Dense: hidden_1 units
        -> ReLU
        -> Dense: hidden_2 units
        -> ReLU
        -> Dense: hidden_3 units
    """

    r=1e-3

    inputs = base_model.input
    x = base_model.output

    projection_1 = Dense(hidden_1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    projection_1 = BatchNormalization()(projection_1)
    projection_1 = Activation("relu")(projection_1)
    # projection_2 = Dense(hidden_2, activation="relu")(projection_1)
    projection_3 = Dense(hidden_3, activation=None,kernel_regularizer=tf.keras.regularizers.l2(l=r))(projection_1)
    projection_3 = BatchNormalization()(projection_3)

    simclr_model = Model(inputs, projection_3, name= base_model.name + "_simclr")

    return simclr_model