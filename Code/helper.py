import numpy as np
from tensorflow.keras import backend as K
import random
import numpy.random as rng
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import mlflow

def scatter(x, labels_str, subtitle=None):
    # We choose a color palette with seaborn.
    le=LabelEncoder()
    labels = le.fit_transform(labels_str)
    palette = np.array(sns.color_palette("hls", np.max(labels)+1))

    # We create a scatter plot.
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=5,
                    c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(np.max(labels)+1):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=14)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
        
    if subtitle != None:
        plt.suptitle(subtitle)
        
    return fig

def __randint_unequal(lower, upper):
        """
        Get two random integers that are not equal.

        Note: In some cases (such as there being only one sample of a class) there may be an endless loop here. This
        will only happen on fairly exotic datasets though. May have to address in future.
        :param lower: Lower limit inclusive of the random integer.
        :param upper: Upper limit inclusive of the random integer. Need to use -1 for random indices.
        :return: Tuple of (integer, integer)
        """
        int_1 = random.randint(lower, upper)
        int_2 = random.randint(lower, upper)
        while int_1 == int_2:
            int_1 = random.randint(lower, upper)
            int_2 = random.randint(lower, upper)
        return int_1, int_2

def class_separation(y_train):
    class_idxs=[]
    for data_class in sorted(set(y_train)):
        class_idxs.append(np.where((y_train == data_class))[0])
    return class_idxs

def euclidean_loss(y_true, y_pred):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """

    loss=y_true*K.square(y_pred)+(1-y_true)*K.square(K.maximum(5-y_pred,0))

    return loss

def get_batch(batch_size,x_train,y_train,idxs_per_class):
    """Create batch of n pairs, half same class, half different class"""
    n_classes=len(idxs_per_class)
    # randomly sample classes to use in the batch
    categories = rng.choice(n_classes,size=(batch_size,),replace=True)
    # pairs1 have the anchors while pairs2 is either positive or negative
    pairs1=[]
    pairs2=[]
    # initialize vector for the targets
    targets=np.zeros((batch_size,),dtype='float')
    # make lower half of it '1's, so 2nd half of batch has same class
    targets[batch_size//2:] = 1.0
    for i in range(batch_size):
        category = categories[i]
        if i>=batch_size//2: #positive
            idx = rng.choice(len(idxs_per_class[category]),size=(2,),replace=False)
            pairs1.append(x_train[idxs_per_class[category][idx[0]]])
            pairs2.append(x_train[idxs_per_class[category][idx[1]]])
        else: #negative
            category2=(category+rng.randint(1,n_classes-1))%n_classes #pick from a different class
            idx1 = rng.randint(0, len(idxs_per_class[category]))
            idx2 = rng.randint(0, len(idxs_per_class[category2]))
            pairs1.append(x_train[idxs_per_class[category][idx1]])
            pairs2.append(x_train[idxs_per_class[category2][idx2]])
    return pairs1, pairs2, targets

def nearest_centroid_NCM(data,train_embeds,calibration_embeds,num_classes,embeddings_size):
    centroids=np.empty((num_classes,embeddings_size))
    for i in range(num_classes):
        centroids[i]=np.mean(train_embeds[data['y_train']==i],axis=0)

    calibration_nc=np.empty(len(data['y_validation']))
    temp_distances=np.zeros(num_classes)
    for i in range(len(data['y_validation'])):
        for j in range(num_classes):
            temp_distances[j]=np.linalg.norm(calibration_embeds[i]-centroids[j])
        calibration_nc[i]=temp_distances[data['y_validation'][i]]/np.min(temp_distances[np.arange(len(temp_distances))!=data['y_validation'][i]])
    return calibration_nc,centroids

def compute_pvalues(data,calibration_nc,centroids,test_embeds):
    num_classes,embeddings_size=centroids.shape
    p_values=np.empty((len(data['y_test']),num_classes))
    centroid_distances=np.zeros(num_classes)
    for i in range(len(data['y_test'])):
        for j in range(num_classes):
            centroid_distances[j]=np.linalg.norm(test_embeds[i]-centroids[j])
        for j in range(num_classes):
            temp_nc=centroid_distances[j]/np.min(centroid_distances[np.arange(len(centroid_distances))!=j])
            p_values[i,j]=np.count_nonzero(calibration_nc>=temp_nc)/len(calibration_nc)
    return p_values

def efficiency_calibration(p_values,y,epsilon):
    mult=0
    error=0
    for i in range(p_values.shape[0]):
        if np.count_nonzero(p_values[i]>=epsilon)>1:
            mult+=1
        if p_values[i][y[i]]<epsilon:
            error+=1
    return mult/p_values.shape[0],error/p_values.shape[0]

def plot_efficiency_calibration(p_values,y,e_start,e_step,e_end):
    sig_levels=np.arange(e_start,e_end,e_step)
    perf_hist=np.empty(len(sig_levels))
    cal_hist=np.empty(len(sig_levels))
    for i,eps in enumerate(sig_levels):
        perf_hist[i],cal_hist[i]=efficiency_calibration(p_values,y,eps)

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot()
    ax.grid(True,axis='both')
    ax.plot(sig_levels,perf_hist, label='Performance')
    ax.plot(sig_levels,cal_hist, label='Calibration')
    ax.plot([0,e_end],[0,e_end],'k--')
    ax.set_xlabel("Significance Level",fontsize=14)
    ax.set_ylabel("Performance & Calibration",fontsize=14)
    ax.legend(loc='upper right')
    fig.tight_layout()
    mlflow.log_figure(fig, "performance_calibration.png")

    ece=np.mean(np.abs(sig_levels-cal_hist))
    mce=np.max(np.abs(sig_levels-cal_hist))
    mlflow.log_metric('ece',ece)
    mlflow.log_metric('mce',mce)
