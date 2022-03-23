import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv1D,LSTM

# The structure of the encoder
def model_LSTM(input_shape1=[128,2],classes=11):
    dr=0.3
    r=1e-4

    input1=Input(input_shape1,name='I/Qchannel')
    x = input1

    x = Conv1D(32, 24,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x = LSTM(units=128,return_sequences=True,name="LSTM1",kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x = tf.keras.layers.Dropout(dr)(x)
    x = LSTM(units=128,return_sequences=True,name="LSTM2",kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x = tf.keras.layers.Dropout(dr)(x)
    x = Conv1D(128, 8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(x)
    model=Model(inputs=input1,outputs=x)
    
    return model
