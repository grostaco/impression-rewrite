import numpy as np
import pickle


def make_prediction(model, encoder, text: str, top_n=3):
    pred = model([text])
    idx = np.argsort(-pred[0])[:3]
    for i in idx:
        mask = np.zeros(shape=(6,))
        mask[i] = 1
        yield (encoder.inverse_transform([mask])[0][0], pred[0][i])


def load_encoder(path: str):
    return pickle.load(open(path, 'rb'))
