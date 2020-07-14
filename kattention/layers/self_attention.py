from tensorflow.keras.layers import Layer, Dense, Dot, Softmax


class SelfAttention(Layer):

    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.dot_context = Dot(axes=(2, 2))
        self.normalize = Softmax()
        self.reweight_v = Dot(axes=(2, 1))

    def build(self, input_shape):
        self.embedding_size = input_shape[-1]
        self.dense_q = Dense(units=self.embedding_size, activation='linear')
        self.dense_k = Dense(units=self.embedding_size, activation='linear')
        self.dense_v = Dense(units=self.embedding_size, activation='linear')

    def call(self, inputs, **kwargs):
        queries = self.dense_q(inputs)
        keys = self.dense_k(inputs)
        values = self.dense_v(inputs)
        context_weights = self.dot_context([queries, keys])
        normalized_context_weights = self.normalize(context_weights)
        return self.reweight_v([normalized_context_weights, values])
