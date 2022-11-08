import tensorflow_text as _  # required for bert model
import tensorflow as tf
import tensorflow_hub as hub


def create_model(tfhub_handle_preprocess, tfhub_handle_encoder):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(
        tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder,
                             trainable=True, name='bert_encoder')
    outputs = encoder(encoder_inputs)

    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(
        6, activation='softmax', name='multi_classifier')(net)
    return tf.keras.Model(text_input, net)


def load_model(path: str):
    return tf.saved_model.load(path)


def train_model(model, export_path: str):
    ...
