

import os
import tensorflow as tf

def preprocess_images():
    ANC_PATH = os.path.join("data", "anchor")
    POS_PATH = os.path.join("data", "positive")
    NEG_PATH = os.path.join("data", "negative")

    def preprocess(file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img

    anchor = tf.data.Dataset.list_files(os.path.join(ANC_PATH, "*.jpg")).take(300)
    positive = tf.data.Dataset.list_files(os.path.join(POS_PATH, "*.jpg")).take(300)
    negative = tf.data.Dataset.list_files(os.path.join(NEG_PATH, "*.jpg")).take(300)

    positives = tf.data.Dataset.zip(
        (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor))))
    )
    negatives = tf.data.Dataset.zip(
        (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor))))
    )
    data = positives.concatenate(negatives)

    def preprocess_twin(input_img, validation_img, label):
        return (preprocess(input_img), preprocess(validation_img), label)

    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=1024)

    train_data = data.take(round(len(data) * 0.7))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    test_data = data.skip(round(len(data) * 0.7))
    test_data = test_data.take(round(len(data) * 0.3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)

    return train_data, test_data
