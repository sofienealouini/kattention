# Kattention

This package implements different Attention mechanisms as Keras layers.

## Setup

```
pip install kattention
```

## Usage

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Softmax
from kattention.layers import MultiHeadAttention 

SEQUENCE_LENGTH = 20
EMBEDDING_SIZE = 300
CLASSES_TO_PREDICT = 4

model = Sequential()
model.add(MultiHeadAttention(heads=5, input_shape=(SEQUENCE_LENGTH, EMBEDDING_SIZE)))
model.add(MultiHeadAttention(heads=5))
model.add(Flatten())
model.add(Dense(CLASSES_TO_PREDICT))
model.add(Softmax())

print(model.summary())
```
