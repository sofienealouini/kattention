from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


def plot_custom_layer(layer: Layer, input_shape: tuple, plot_filepath: str):
    tmp_input = Input(shape=input_shape)
    layer.build(input_shape)
    tmp_output = layer.call(tmp_input)
    tmp_model = Model(inputs=[tmp_input], outputs=[tmp_output])
    return plot_model(tmp_model, to_file=plot_filepath, show_shapes=True)
