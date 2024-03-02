import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Flatten
from keras import Model
import keras
import warnings

warnings.filterwarnings("ignore")



def build_embedding_generator(k_layers_to_tune=10):

    base_model = tf.keras.applications.EfficientNetB1(weights="imagenet", 
                                                      input_shape=(100, 100, 3),
                                                      include_top = False)

    for l in base_model.layers[:-k_layers_to_tune]:
        l.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="linear")(x)
    x = tf.nn.l2_normalize(x, axis=1)


    
    embedding_model = Model(base_model.input, x, name="Embedding")

    return embedding_model


class DistanceLayer(tf.keras.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):

        anchor_pos_distance = tf.reduce_sum(tf.square(anchor - positive))
        anchor_neg_distance = tf.reduce_sum(tf.square(anchor - negative))

        return (anchor_pos_distance, anchor_neg_distance)
    
    
class CosineSimilarityLayer(tf.keras.layers.Layer):
    """
    This layer is responsible for computing the cosine similarity between
    the anchor embedding and the positive embedding, and the anchor embedding
    and the negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        # Normalize the vectors to unit length
        anchor_normalized = tf.nn.l2_normalize(anchor, axis=1)
        positive_normalized = tf.nn.l2_normalize(positive, axis=1)
        negative_normalized = tf.nn.l2_normalize(negative, axis=1)

        # Cosine similarity = dot product of normalized vectors
        pos_similarity = tf.reduce_sum(tf.multiply(anchor_normalized, positive_normalized), axis=1)
        neg_similarity = tf.reduce_sum(tf.multiply(anchor_normalized, negative_normalized), axis=1)

        return pos_similarity, neg_similarity


def build_siamesenetwork(embedding_model):

    anchor_input = keras.layers.Input(name="anchor", shape=(100, 100, 3))
    pos_input = keras.layers.Input(name="positive", shape=(100, 100, 3))
    neg_input = keras.layers.Input(name="negative", shape=(100, 100, 3))

    distances = DistanceLayer()(
        embedding_model(anchor_input),
        embedding_model(pos_input),
        embedding_model(neg_input)
    )

#     distances = CosineSimilarityLayer()(
        
#         embedding_model(anchor_input),
#         embedding_model(pos_input),
#         embedding_model(neg_input)  ) 

    siamese_network = Model(
            inputs=[anchor_input, pos_input, neg_input],
            outputs=distances
    )

    return siamese_network





class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """
    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    

    def train_step(self, data):

        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))
        
        self.loss_tracker.update_state(loss)

        return {"loss" : self.loss_tracker.result()}

    def _compute_loss(self, data):

        ap_distance, an_distance = self.siamese_network(data)

        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss
     
            
    def test_step(self, data):
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)

        return {"loss" : self.loss_tracker.result()}

    @property
    def metrics(self):

        return [self.loss_tracker]
