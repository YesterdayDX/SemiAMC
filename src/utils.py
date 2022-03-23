import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.lstm import *
from src.simclr_model import attach_simclr_head
from src.simclr_utility import *
from src.data_aug import *
from tensorflow.keras.layers import Dense,Dropout

def normalize_data(X_train, X_train_labeled, X_val_labeled, X_test):
    # Normalize the data
    for i in range(X_train.shape[0]):
        m = np.max(np.absolute(X_train[i]))
        X_train[i] = X_train[i]/m
    for i in range(X_train_labeled.shape[0]):
        m = np.max(np.absolute(X_train_labeled[i]))
        X_train_labeled[i] = X_train_labeled[i]/m
    for i in range(X_test.shape[0]):
        m = np.max(np.absolute(X_test[i]))
        X_test[i] = X_test[i]/m
    for i in range(X_val_labeled.shape[0]):
        m = np.max(np.absolute(X_val_labeled[i]))
        X_val_labeled[i] = X_val_labeled[i]/m   
    return X_train, X_train_labeled, X_val_labeled, X_test

def train_supervised(X_train_labeled, Y_train_labeled, X_val_labeled, Y_val_labeled, X_test, Y_test, batch_size=512, Epoch=500):

    encoder = model_LSTM(classes=11)
    inputs = encoder.inputs

    dr=0.3
    r=1e-4
    x = encoder.output
    x=Dropout(dr)(x)
    x=Dense(128,activation="selu",name="FC1",kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x=Dropout(dr)(x)

    outputs = Dense(11, activation="softmax", name="linear_Classifier")(x)
    sup_model = Model(inputs=inputs, outputs=outputs, name="Sup_Model")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    sup_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    sup_model.summary()

    history = sup_model.fit(X_train_labeled,
        Y_train_labeled,
        batch_size=batch_size,
        epochs=Epoch,
        verbose=2,
        validation_data=(X_val_labeled,Y_val_labeled),
        callbacks = [
                    tf.keras.callbacks.ModelCheckpoint("./saved_models/weight_sup.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.8,verbose=1,patince=5,min_lr=0.0000001),
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto'),           
                    ]
                        )

    print("=========== Supervised Training Completed! ==========")
    print("Save model to \'saved_models/weight_sup.hdf5\'")

    return sup_model

def train_tune(X_train_labeled, Y_train_labeled, X_val_labeled, Y_val_labeled, X_test, Y_test, batch_size=512, Epoch=200):
    # Load sim_model from file
    sim_model = tf.keras.models.load_model("./saved_models/weight_simclr.hdf5")
    
    inputs = sim_model.inputs
    x = sim_model.layers[-6].output

    dr=0.3
    r=1e-4

    x=Dropout(dr,name="DP_1")(x)
    x=Dense(128,activation="selu",name="FC1",kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x=Dropout(dr,name="DP_2")(x)
    # x=Dense(128,activation="selu",name="FC2",kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)

    outputs = Dense(11, activation="softmax", name="linear_Classifier")(x)
    tune_model = Model(inputs=inputs, outputs=outputs, name="Tune_Model")
    
    for layer in tune_model.layers:
        layer.trainable = False

    # We can choose # of layers to be tuned according to the amount of labeled training data we have.
    # We tune less layers when we have a smaller number of labeled data.
    for layer in tune_model.layers[-8:]:
        layer.trainable = True

    tune_model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    tune_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    training_history = tune_model.fit(
        x = X_train_labeled,
        y = Y_train_labeled,
        batch_size=batch_size,
        shuffle=True,
        epochs=Epoch,
        verbose=2,
        callbacks = [
                    tf.keras.callbacks.ModelCheckpoint("./saved_models/weight_tune.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.8,verbose=1,patince=5,min_lr=0.0000001),
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto'),           
                    ],
        validation_data=(X_val_labeled,Y_val_labeled)
    )

    print("=========== Tuning Completed! ==========")
    print("Save model to \'saved_models/weight_tune.hdf5\'")

    return tune_model

def train_simclr(X_train, batch_size=512, Epoch=100, temperature = 0.1):
    decay_steps = 15000

    # Training
    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.001, decay_steps=decay_steps)
    optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)

    base_model = model_LSTM(classes=11)
    sim_model = attach_simclr_head(base_model)
    sim_model.summary()
    trained_simclr_model, epoch_losses = simclr_train_model(sim_model, X_train, optimizer, batch_size, temperature=temperature, epochs=Epoch, verbose=1)

    print("=========== Contrastive Training Completed! ==========")

    # Save Training Loss
    np.savetxt('./results/epoch_loss.csv', epoch_losses)
    return trained_simclr_model, epoch_losses

def plot_epoch_loss(loss_file='./results/epoch_loss.csv'):
    '''
    Plot the loss for each epoch during contrastive self-supervised training
    '''
    losses = np.loadtxt(loss_file)
    plt.figure()
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.title("Epoch Loss")
    plt.plot(losses)
    plt.savefig('./results/epoch_loss.png')

def compare_tune_and_sup(weight_tune, weight_sup, X_test, Y_test, test_idx, snrs, lbl, batch_size=512):
    tune_model = tf.keras.models.load_model(weight_tune)
    sup_model = tf.keras.models.load_model(weight_sup)

    score = tune_model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print("Tuned model score:", score)
    score = sup_model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print("Supervised model score:", score)

    acc_tune = {}
    acc_sup = {}

    for snr in snrs:
        # Extract classes @ SNR
        test_SNRs = [lbl[x][1] for x in test_idx]
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
        
        # # Estimate classes
        test_hat_i_tune = tune_model.predict(test_X_i)
        test_hat_i_sup = sup_model.predict(test_X_i)

        label_i= np.argmax(test_Y_i, axis=1)
        pred_i_tune = np.argmax(test_hat_i_tune, axis=1)
        pred_i_sup = np.argmax(test_hat_i_sup, axis=1)

        acc_tune[snr] = 1.0 * np.mean(label_i==pred_i_tune)
        acc_sup[snr] = 1.0 * np.mean(label_i==pred_i_sup)
        
    # Plot accuracy curve
    plt.figure()
    plt.plot(snrs, list(map(lambda x: acc_tune[x], snrs)), 'd-', label="SemiAMC")
    plt.plot(snrs, list(map(lambda x: acc_sup[x], snrs)), 'o-', label="Supervised")
    plt.xlim([-20,20])
    plt.ylim([0,0.8])
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title(" Classification Accuracy on RadioML")
    plt.grid('-.')
    plt.tight_layout()
    plt.legend()
    plt.savefig('./results/acc_under_snrs.png')