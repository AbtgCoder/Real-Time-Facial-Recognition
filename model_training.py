# model_training.py

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def make_embedding():
    inp = Input(shape=(100, 100, 3))
    c1 = Conv2D(64, (10, 10), activation="relu")(inp)
    m1 = MaxPooling2D(64, (2, 2), padding="same")(c1)
    c2 = Conv2D(128, (7, 7), activation="relu")(m1)
    m2 = MaxPooling2D(64, (2, 2), padding="same")(c2)
    c3 = Conv2D(128, (4, 4), activation="relu")(m2)
    m3 = MaxPooling2D(64, (2, 2), padding="same")(c3)
    c4 = Conv2D(256, (4, 4), activation="relu")(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation="sigmoid")(f1)
    return Model(inputs=[inp], outputs=[d1], name="embedding")


def make_siamese_model(embedding_model):
    input_image = Input(name="input_img", shape=(100, 100, 3))
    validation_image = Input(name="validation_img", shape=(100, 100, 3))
    siamese_layer = L1Dist()
    siamese_layer._name = "distance"
    distances = siamese_layer(embedding_model(input_image), embedding_model(validation_image))
    classifier = Dense(1, activation="sigmoid")(distances)
    return Model(inputs=[input_image, validation_image], outputs=classifier, name="SiameseNetwork")


def train_model(train_data, EPOCHS):
    embedding_model = make_embedding()
    siamese_model = make_siamese_model(embedding_model)
    binary_cross_loss = tf.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(1e-4)
    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            X = batch[:2]
            y = batch[2]
            yhat = siamese_model(X, training=True)
            loss = binary_cross_loss(y, yhat)

        grad = tape.gradient(loss, siamese_model.trainable_variables)
        opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
        return loss

    def train(data, EPOCHS):
        for epoch in range(1, EPOCHS + 1):
            print(f"\n Epoch {epoch}/{EPOCHS}")
            progbar = tf.keras.utils.Progbar(len(data))
            for idx, batch in enumerate(data):
                train_step(batch)
                progbar.update(idx + 1)
            if epoch % 10 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

    train(train_data, EPOCHS)
    siamese_model.save("siamese_model.h5")
    return siamese_model

