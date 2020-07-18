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
from kattention.layers import Transformer 

SEQUENCE_LENGTH = 4
EMBEDDING_SIZE = 300
CLASSES_TO_PREDICT = 5
ATT_HEADS = 2

model = Sequential()
model.add(Transformer(attention_heads=ATT_HEADS, input_shape=(SEQUENCE_LENGTH, EMBEDDING_SIZE)))
model.add(Transformer(attention_heads=ATT_HEADS))
model.add(Flatten())
model.add(Dense(CLASSES_TO_PREDICT))
model.add(Softmax())

print(model.summary())
```
