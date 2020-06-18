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

IMAGE_SIZE = [256, 256, 3]


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    This triplet loss function is specified for the keras model. 
    It uses the l-2 norm to quantify the distance between the (anchor-positive) and (anchor-negative) vectors

    It is based on the following article:
    <https://medium.com/@prabhnoor0212/siamese-network-keras-31a3a8f37d04>


    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
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
        if layer_count < len(pretrained_basemodel.layers) - 4:
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
    Train the model
    ═══════════════

    Arguments
    ─────────

    • model: the model to be trained
    • epochs_count: number of epochs used for training.
    one epoch corresponds to passing all triplets from the training set
    once thourgh the network.
    • backup: Save a backup of the model after every epoch. (defaults to
        true)
    """
    
    model = tf.keras.models.load_model("saved_model/current_model.h5", custom_objects={'triplet_loss':triplet_loss}, compile=True)
    for i in range(epochs_count):
        for k in range(batch_count):
            print('working on batch: ', k)
            model.fit(fm.return_training_data(batch_size, k, IMAGE_SIZE),
                    epochs=1,
                    verbose=1)
            model.save(f"saved_model/current_model.h5")

def predict_test_triplets(model, batch_size, batch_count):
    classifications = np.array([], dtype=np.bool)
    for k in tqdm(range(batch_count)):
        test_set = fm.return_test_data(batch_size, IMAGE_SIZE, k)
        batch_prediction = model.predict(test_set, verbose=1)
        classifications = np.append(classifications, classify_prediction(batch_prediction))
    np.savetxt(f"test_set_predictions.txt", classifications, fmt="%i")

def classify_prediction(prediction):
    """
    Classification
    ══════════════

    The prediction embedding (output of the model) is analyzed to
    determine whether the first image is closer to the second or the
    third.


    Arguments
    ─────────

    • prediction: embedding of the inputs.
        • size: (batch_size, embedding_size*3)


    Returns
    ───────

    Tensor of dimension (batch_size), result of classification
    """
    total_length = prediction.shape[-1]
    anchor = prediction[:, 0:int(total_length/3)]
    positive = prediction[:, int(total_length/3):int(total_length*2/3)]
    negative = prediction[:, int(total_length*2/3):total_length]

    pos_dist = tf.keras.backend.sum(tf.keras.backend.square(anchor-positive), axis=1)
    neg_dist = tf.keras.backend.sum(tf.keras.backend.square(anchor-negative), axis=1)

    return tf.keras.backend.less_equal(pos_dist, neg_dist)


def main():
    # fm.resize_images(IMAGE_SIZE[0], IMAGE_SIZE[1])
    model = create_model()
    train_model(model, 1, 10, 1)
    predict_test_triplets(model, 10, 8)

    return



if __name__ == "__main__":

    main()