"Adapted from the code (https://github.com/leena201818/radioml) contributed by leena201818"
import pickle
import numpy as np
import math
   
def load_data(filename, size_train_labeled, size_val_labeled):
    '''
    RadioML2016.10a: (220000,2,128), mods*snr*1000, total 220000 samples

    There are 1000 samples per snr for each modulation type, we use 750 as training set, 250 as validation set,
    and 250 as testing set. We use all data in the training and validation set (without label) to do the 
    contrastive learning. Then, among the training and validation sets, we choose 'size_train_labeled' and 
    'size_val_labeled' data samples as labeled to tune the classifier. 

    Input: 
        filename: path of dataset
        size_train_labeled: [0,1000], # of labeled training samples
        size_val_labeled: [0,1000], # of labeled validation samples
    Output:
        X_train, Y_train: all training data set
        X_val, Y_val: all validation data set
        X_test, Y_test: all testing data set
        X_train_labeled, Y_train_labeled: labeled training data
        X_val_labeled, Y_val_labeled: labeled validation data
    '''
    # np.random.seed(1991)

    # size_train_labeled = round(rate * 500 / 100) # size_train_labeled: [0,1000], # of labeled training samples
    # size_val_labeled = round(math.ceil(size_train_labeled/2.0)) # size_val_labeled: [0,1000], # of labeled training samples

    # RadioML2016.10a: (220000,2,128), mods*snr*1000, total 220000 samples; 
    Xd =pickle.load(open(filename,'rb'),encoding='iso-8859-1')
    mods,snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1] ] 
    X = []
    lbl = []
    train_idx=[]
    val_idx=[]
    train_labeled_idx=[]
    # train_unlabel_idx=[]
    val_labeled_idx=[]
    np.random.seed(2016)
    a=0

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)]) 
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
            
            inds_train = list(np.random.choice(range(a*1000,(a+1)*1000), size=500, replace=False))
            train_idx+=inds_train
            inds_val = list(np.random.choice(list(set(range(a*1000,(a+1)*1000))-set(train_idx)), size=250, replace=False))
            val_idx+=inds_val

            inds_train_labeled = list(np.random.choice(inds_train, size=size_train_labeled, replace=False))
            inds_val_labeled = list(np.random.choice(inds_val, size=size_val_labeled, replace=False))

            train_labeled_idx+=inds_train_labeled
            val_labeled_idx+=inds_val_labeled
            a+=1

    X = np.vstack(X) 
    X = np.swapaxes(X,1,2)

    # Scramble the order between samples 
    # and get the serial number of training, validation, and test sets
    n_examples=X.shape[0]

    test_idx=list(set(range(0,n_examples))-set(train_idx)-set(val_idx))
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(train_labeled_idx)
    np.random.shuffle(val_labeled_idx)
    np.random.shuffle(test_idx)

    train_idx = np.concatenate((train_idx, val_idx), axis=0)

    X_train =X[train_idx]
    # X_val=X[val_idx]
    X_train_labeled =X[train_labeled_idx]
    X_val_labeled=X[val_labeled_idx]
    X_test =X[test_idx]

    # transfor the label form to one-hot
    def to_onehot(yy):
        yy1=np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy]=1
        return yy1

    Y_train=to_onehot(list(map(lambda x: mods.index(lbl[x][0]),train_idx)))
    # Y_val=to_onehot(list(map(lambda x: mods.index(lbl[x][0]),val_idx)))
    Y_train_labeled=to_onehot(list(map(lambda x: mods.index(lbl[x][0]),train_labeled_idx)))
    Y_val_labeled=to_onehot(list(map(lambda x: mods.index(lbl[x][0]),val_labeled_idx)))
    Y_test=to_onehot(list(map(lambda x: mods.index(lbl[x][0]),test_idx)))

    return (mods,snrs,lbl),(X_train,Y_train),(X_train_labeled,Y_train_labeled),(X_val_labeled,Y_val_labeled),(X_test,Y_test),(train_idx,test_idx,train_labeled_idx,val_labeled_idx)

