# scripts/deepLIFT_attribution.py

import numpy as np
import tensorflow as tf
from deeplift.dinuc_shuffle import dinuc_shuffle
from deeplift.layers import NonlinearMxtsMode
from deeplift.util import compile_func
import deeplift.layers as deeplift_layers
from deeplift.models import Sequential

def get_deeplift_model(keras_model):
    deeplift_model = Sequential()
    for layer in keras_model.layers:
        deeplift_layer = deeplift_layers.Conv1D(layer, mode=NonlinearMxtsMode.DeepLIFT)
        deeplift_model.add(deeplift_layer)
    return deeplift_model

def calculate_attributions(deeplift_model, X_val):
    deeplift_func = compile_func(
        deeplift_model.get_target_contribs_func(find_scores_layer_idx=0, target_layer_idx=-1),
        input_layer_idx=0,
        batch_size=10
    )
    attributions = deeplift_func(task_idx=0, input_data_list=[X_val], input_references_list=[X_val.mean(axis=0, keepdims=True)])
    return attributions

def main():
    X_val = np.load('data/processed/X_val.npy')
    model = tf.keras.models.load_model('results/model/bpnet_model.h5')

    deeplift_model = get_deeplift_model(model)
    attributions = calculate_attributions(deeplift_model, X_val)

    np.save('results/attributions/attributions.npy', attributions)

if __name__ == "__main__":
    main()

