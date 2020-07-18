from tensorflow.keras.layers import Layer, Dense, Dot, Softmax
from tensorflow.python.keras.layers import Lambda


class SelfAttention(Layer):

    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.dense_q = None
        self.dense_k = None
        self.dense_v = None
        self.dot_context = Dot(axes=(2, 2))
        self.scale = None
        self.normalize = Softmax()
        self.reweight_v = Dot(axes=(2, 1))

    def build(self, input_shape):
        embedding_size = input_shape[-1]
        self.dense_q = Dense(units=embedding_size, activation='linear')
        self.dense_k = Dense(units=embedding_size, activation='linear')
        self.dense_v = Dense(units=embedding_size, activation='linear')
        scaling_factor = 1 / embedding_size
        self.scale = Lambda(lambda x: x * scaling_factor)

    def call(self, inputs, **kwargs):
        queries = self.dense_q(inputs)
        keys = self.dense_k(inputs)
        values = self.dense_v(inputs)
        context_weights = self.dot_context([queries, keys])
        scaled_context_weights = self.scale(context_weights)
        normalized_context_weights = self.normalize(scaled_context_weights)
        return self.reweight_v([normalized_context_weights, values])
