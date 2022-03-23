import numpy as np

def rotation_2d(x, ang=90):
    x_aug = np.empty(x.shape)
    if ang==0:
        x_aug = x
    elif ang==90:
        x_aug[:,0] = -x[:,1]
        x_aug[:,1] = x[:,0]
    elif ang==180:
        x_aug = -x
    elif ang==270:
        x_aug[:,0]=x[:,1]
        x_aug[:,1]=-x[:,0]
    else:
        print("Wrong input for rotation!")
    return x_aug

def data_aug_rotation(X_batch):
    angs = [0,90,180,270]

    X_aug1 = []
    X_aug2 = []
    [ang1, ang2] = np.random.choice(angs,2,replace=False)

    for x in X_batch:
        x_aug1 = rotation_2d(x, ang1)
        x_aug2 = rotation_2d(x, ang2)
        X_aug1.append(x_aug1)
        X_aug2.append(x_aug2)
    return np.array(X_aug1), np.array(X_aug2)