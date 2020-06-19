#    _____         .__
#   /     \ _____  |__| ____
#  /  \ /  \\__  \ |  |/    \
# /    Y    \/ __ \|  |   |  \
# \____|__  (____  /__|___|  /
#         \/     \/        \/
'''The main file is used to start up the program'''

__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "GPL"

import numpy as np
import pandas as pd
import tensorflow as tf
import file_manager as fm
from tqdm import tqdm
import datetime

IMAGE_SIZE = [256, 256, 3]


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    This triplet loss function is specified for the keras model. 
    It uses the l-2 norm to quantify the distance between the (anchor-positive) and (anchor-negative) vectors

    It is based on the following article:
    <https://medium.com/@prabhnoor0212/siamese-network-keras-31a3a8f37d04>

    Inputs:
    y_true -- true labels
            anchor_img -- the encodings for the anchor data
            positive_img -- the encodings for the positive data (similar to anchor)
            negative_img -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ',y_pred)
    
    total_lenght = y_pred.shape.as_list()[-1]
    print(total_lenght)
    
    anchor_img = y_pred[:,0:int(total_lenght*1/3)]
    positive_img = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    negative_img = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

    # distance between the anchor and the positive
    pos_dist = tf.keras.backend.sum(tf.keras.backend.square(anchor_img-positive_img), axis=1)

    # distance between the anchor and the negative
    neg_dist = tf.keras.backend.sum(tf.keras.backend.square(anchor_img-negative_img), axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = tf.keras.backend.maximum(basic_loss, 0.0)
    return loss
    

def create_model():
    """
    Creates a Keras model containing the CNN based on the pretrained ResNet50 model
    """
    # initializing the pretrained model
    pretrained_basemodel = tf.keras.applications.ResNet50(include_top=False)

    # only enable learning for the last three layers
    for layer_count, layer in enumerate(pretrained_basemodel.layers):
        if layer_count < len(pretrained_basemodel.layers) - 8:
            layer.trainable = False
        else:
            layer.trainable = True

    '''Define the inputs of the three images: The anchor image is the baseline we want to compare to the two other images. 
    the positive image is similar to the baseline whyle the negative image is a different one'''
    anchor_img = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]))
    positive_img = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]))
    negative_img = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]))

    model_emb = tf.keras.models.Sequential()
    model_emb.add(pretrained_basemodel)
    model_emb.add(tf.keras.layers.GlobalMaxPool2D())
    model_emb.add(tf.keras.layers.Flatten()) 
    model_emb.add(tf.keras.layers.Dense(64,
                                              activation=None,
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                              kernel_initializer="he_uniform"))
    model_emb.add(tf.keras.layers.Lambda(lambda x : tf.keras.backend.l2_normalize(x)))

    a = model_emb(anchor_img)
    p = model_emb(positive_img)
    n = model_emb(negative_img)

    apn = tf.keras.backend.concatenate([a, p, n], axis=-1)
    model = tf.keras.models.Model([anchor_img, positive_img, negative_img], apn)
    model.compile(loss=triplet_loss, optimizer=tf.keras.optimizers.Adam(0.00015))
    model.summary()
    return model

def train_model(model, epochs_count, batch_size, batch_count, backup=True):
    """
    used to train the model with data.

    Inputs:

    - model: the model to be trained
    - epochs_count: the number of full cycles trough the training data
    - batch_size: size of the batch that is passed for training at once
    - batch_count: the amount consequtive training batches. can be maximum: sample_size/batchsize rounded up
    """
    print(datetime.datetime.now())
    model = tf.keras.models.load_model("saved_model/current_model.h5", custom_objects={'triplet_loss':triplet_loss}, compile=True)
    # for i in range(epochs_count):
    for k in tqdm(range(batch_count)):
        print('working on batch: ', k)
        model.fit(fm.return_training_data(batch_size, k, IMAGE_SIZE),
                epochs=epochs_count,
                verbose=1)
        model.save(f"saved_model/current_model.h5")

def predict_test_triplets(model, batch_size, batch_count):
    """
    used to predict classifications for the test set.

    Inputs:

    - model: the model to be trained
    - batch_size: size of the batch that is passed for training at once
    - batch_count: the amount consequtive training batches. can be maximum: sample_size/batchsize rounded up
    """
    classifications = np.array([], dtype=np.bool)

    for k in tqdm(range(batch_count)):
        test_set = fm.return_test_data(batch_size, IMAGE_SIZE, k)
        batch_prediction = model.predict(test_set, verbose=1)
        classifications = np.append(classifications, classify_prediction(batch_prediction))
    np.savetxt(f"test_set_predictions_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + ".txt", classifications, fmt="%i")

def classify_prediction(prediction):
    """
    this function is used to determine which if the images B or C is closer to A. 

    Inputs:

    - prediction: anumpy array containing the predicted feature vectors for a single or multiple triplets. 

    Output:

    - 1 if image b is closer
    - 0 if image c is closer

    """
    prediction_length = prediction.shape[-1]
    anchor = prediction[:, 0:int(prediction_length/3)]
    img_b = prediction[:, int(prediction_length/3):int(prediction_length*2/3)]
    img_c = prediction[:, int(prediction_length*2/3):prediction_length*3/3]
    dist_to_b = tf.keras.backend.sum(tf.keras.backend.square(anchor-img_b), axis=1)
    dist_to_c = tf.keras.backend.sum(tf.keras.backend.square(anchor-img_c), axis=1)
    return tf.keras.backend.less_equal(posistive_dist, negative_dist)


def main():
    # fm.resize_images(IMAGE_SIZE[0], IMAGE_SIZE[1])
    model = create_model()
    train_model(model, 6, 16, 3720)
    predict_test_triplets(model, 1000, 60)

    return



if __name__ == "__main__":

    main()