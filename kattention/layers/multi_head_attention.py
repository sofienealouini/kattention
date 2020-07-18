from tensorflow.keras.layers import Layer, Dense, Concatenate

from kattention.layers import SelfAttention


class MultiHeadAttention(Layer):

    def __init__(self, heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.attention_heads = [SelfAttention() for _ in range(self.heads)]
        self.concat_heads = Concatenate()
        self.embed = None

    def build(self, input_shape):
        embedding_size = input_shape[-1]
        self.embed = Dense(units=embedding_size, activation='linear')

    def call(self, inputs, **kwargs):
        self_attention_outputs = [attention_head(inputs) for attention_head in self.attention_heads]
        concatenated_heads = self.concat_heads(self_attention_outputs)
        return self.embed(concatenated_heads)
