"Adapted from the code (https://github.com/iantangc/ContrastiveLearningHAR) contributed by iantangc"
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,BatchNormalization,Activation
from src.data_aug import *


def attach_simclr_head(base_model, hidden_1=128, hidden_2=64):
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
    projection_2 = Dense(hidden_2, activation=None,kernel_regularizer=tf.keras.regularizers.l2(l=r))(projection_1)
    projection_2 = BatchNormalization()(projection_2)

    simclr_model = Model(inputs, projection_2, name= base_model.name + "_simclr")

    return simclr_model


def simclr_train_model(model, dataset, optimizer, batch_size, temperature=1.0, epochs=100, verbose=0):
    """
    Train a deep learning model using the SimCLR algorithm

    Parameters:
        model
            the deep learning model for feature learning 

        dataset
            the numpy array for training (no labels)
            the first dimension should be the number of samples
        
        optimizer
            the optimizer for training
            e.g. tf.keras.optimizers.SGD()

        batch_size
            the batch size for mini-batch training

        temperature = 1.0
            hyperparameter of the NT_Xent_loss, the scaling factor of the logits
            (see NT_Xent_loss)
        
        epochs = 100
            number of epochs of training

        verbose = 0
            debug messages are printed if > 0

    Return:
        (model, epoch_wise_loss)
            model
                the trained model
            epoch_wise_loss
                list of epoch losses during training
    """

    epoch_wise_loss = []
    
    if not os.path.exists('./saved_models/simclr'):
        os.mkdir('./saved_models/simclr')

    for epoch in range(epochs):
        step_wise_loss = []

        # Randomly shuffle the dataset
        shuffle_indices = np.arange(dataset.shape[0])
        np.random.shuffle(shuffle_indices)
        shuffled_dataset = dataset[shuffle_indices]

        # Make a batched dataset
        batched_dataset = get_batched_dataset_generator(shuffled_dataset, batch_size)

        for data_batch in batched_dataset:
            # Apply Data Augmentation
            X_1, X_2 = data_aug_rotation(data_batch)

            # Forward propagation
            loss, gradients = get_NT_Xent_loss_gradients(model, X_1, X_2, normalize=True, temperature=temperature, weights=1.0)

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            step_wise_loss.append(loss)

        epoch_wise_loss.append(np.mean(step_wise_loss))
        
        # if verbose > 0:
        #     print("Epoch {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

        if epoch%10==0:
            print("Epoch {} loss: {:.3f}".format(epoch, np.mean(step_wise_loss)))
            model.save("./saved_models/simclr/weight_"+str(epoch)+".hdf5")
            print("Write to \'saved_models/simclr/weight_"+str(epoch)+".hdf5\'")

        model.save("./saved_models/weight_simclr.hdf5")

    return model, epoch_wise_loss


def get_NT_Xent_loss_gradients(model, samples_transform_1, samples_transform_2, normalize=True, temperature=1.0, weights=1.0):
    """
    A wrapper function for the NT_Xent_loss function which facilitates back propagation

    Parameters:
        model
            the deep learning model for feature learning 

        samples_transform_1
            inputs samples subject to transformation 1
        
        samples_transform_2
            inputs samples subject to transformation 2

        normalize = True
            normalise the activations if true

        temperature = 1.0
            hyperparameter, the scaling factor of the logits
            (see NT_Xent_loss)
        
        weights = 1.0
            weights of different samples
            (see NT_Xent_loss)

    Return:
        loss
            the value of the NT_Xent_loss

        gradients
            the gradients for backpropagation
    """
    with tf.GradientTape() as tape:
        hidden_features_transform_1 = model(samples_transform_1)
        hidden_features_transform_2 = model(samples_transform_2)
        loss = NT_Xent_loss(hidden_features_transform_1, hidden_features_transform_2, normalize=normalize, temperature=temperature, weights=weights)

    gradients = tape.gradient(loss, model.trainable_variables)
    return loss, gradients


"""
The following section of this file includes software licensed under the Apache License 2.0, by The SimCLR Authors 2020, modified by C. I. Tang.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""

def NT_Xent_loss(hidden_features_transform_1, hidden_features_transform_2, normalize=True, temperature=1.0, weights=1.0):
    """
    The normalised temperature-scaled cross entropy loss function of SimCLR Contrastive training
    Reference: Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709.
    https://github.com/google-research/simclr/blob/master/objective.py

    Parameters:
        hidden_features_transform_1
            the features (activations) extracted from the inputs after applying transformation 1
            e.g. model(transform_1(X))
        
        hidden_features_transform_2
            the features (activations) extracted from the inputs after applying transformation 2
            e.g. model(transform_2(X))

        normalize = True
            normalise the activations if true

        temperature
            hyperparameter, the scaling factor of the logits
        
        weights
            weights of different samples

    Return:
        loss
            the value of the NT_Xent_loss
    """
    LARGE_NUM = 1e9
    entropy_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    batch_size = tf.shape(hidden_features_transform_1)[0]
    
    h1 = hidden_features_transform_1
    h2 = hidden_features_transform_2
    if normalize:
        h1 = tf.math.l2_normalize(h1, axis=1)
        h2 = tf.math.l2_normalize(h2, axis=1)

    labels = tf.range(batch_size)
    masks = tf.one_hot(tf.range(batch_size), batch_size)
    
    logits_aa = tf.matmul(h1, h1, transpose_b=True) / temperature
    # Suppresses the logit of the repeated sample, which is in the diagonal of logit_aa
    # i.e. the product of h1[x] . h1[x]
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(h2, h2, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(h1, h2, transpose_b=True) / temperature
    logits_ba = tf.matmul(h2, h1, transpose_b=True) / temperature

    
    loss_a = entropy_function(labels, tf.concat([logits_ab, logits_aa], 1), sample_weight=weights)
    loss_b = entropy_function(labels, tf.concat([logits_ba, logits_bb], 1), sample_weight=weights)
    loss = loss_a + loss_b

    return loss


def get_batched_dataset_generator(data, batch_size):
    """
    Create a data batch generator
    Note that the last batch might not be full

    Parameters:
        data
            A numpy array of data

        batch_size
            the (maximum) size of the batches

    Returns:
        generator<numpy array>
            a batch of the data with the same shape except the first dimension, which is now the batch size
    """

    num_bathes = ceiling_division(data.shape[0], batch_size)
    for i in range(num_bathes):
        yield data[i * batch_size : (i + 1) * batch_size]

def ceiling_division(n, d):
    """
    Ceiling integer division
    """
    return -(n // -d)