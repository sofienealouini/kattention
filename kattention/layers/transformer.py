from tensorflow.keras.layers import Layer, Add, LayerNormalization, Dense

from kattention.layers import MultiHeadAttention


class Transformer(Layer):

    def __init__(self, attention_heads, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.attention_heads = attention_heads
        self.attend = MultiHeadAttention(heads=self.attention_heads)
        self.add_1 = Add()
        self.normalize_1 = LayerNormalization()
        self.dense = None
        self.add_2 = Add()
        self.normalize_2 = LayerNormalization()

    def build(self, input_shape):
        embedding_size = input_shape[-1]
        self.dense = Dense(units=embedding_size, activation='linear')

    def call(self, inputs, **kwargs):
        inputs_with_attention = self.attend(inputs)
        inputs_with_attention_and_residual = self.add_1([inputs_with_attention, inputs])
        normalized_embeddings_with_attention = self.normalize_1(inputs_with_attention_and_residual)
        new_embeddings = self.dense(normalized_embeddings_with_attention)
        new_embeddings_with_residual = self.add_2([normalized_embeddings_with_attention, new_embeddings])
        return self.normalize_2(new_embeddings_with_residual)
