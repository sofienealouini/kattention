from tensorflow.keras.layers import Layer, Dense, Concatenate

from kattention.layers import SelfAttention


class MultiHeadAttention(Layer):

    def __init__(self, heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.concat_heads = Concatenate()

    def build(self, input_shape):
        self.embedding_size = input_shape[-1]
        self.attention_heads = [SelfAttention() for _ in range(1, self.heads + 1)]
        self.embed = Dense(units=self.embedding_size, activation='linear')

    def call(self, inputs, **kwargs):
        self_attention_outputs = [att_head(inputs) for att_head in self.attention_heads]
        concatenated_heads = self.concat_heads(self_attention_outputs)
        compressed_embedding = self.embed(concatenated_heads)

        return compressed_embedding
