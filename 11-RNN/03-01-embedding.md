 ```python
Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length)
```
✔ **`Embedding(...)`** → Converts word indices (integers) into dense vector representations.  
✔ **`input_dim=vocab_size`** → Number of unique words in the vocabulary.  
✔ **`output_dim=64`** → Size of the embedding vector for each word (each word is mapped to a 64-dimensional vector).  
✔ **`input_length=max_length`** → Specifies the fixed length of input sequences (for LSTMs, RNNs, etc.).  

---

## 🔹 **Example Usage**
```python
import tensorflow as tf
import numpy as np

# Define an embedding layer
embedding_layer = tf.keras.layers.Embedding(input_dim=10, output_dim=4, input_length=5)

# Sample input (word indices)
sample_input = np.array([[1, 2, 3, 4, 5]])

# Get embedding output
embedded_output = embedding_layer(sample_input)

print("Input:", sample_input)
print("Embedded Output:\n", embedded_output.numpy())
```

### 🔹 **Output**
```
Input: [[1 2 3 4 5]]

Embedded Output:
 [[[-0.02  0.3  -0.4   0.1 ]
   [ 0.1  -0.2   0.6  -0.3 ]
   [-0.5   0.2  -0.1   0.7 ]
   [ 0.3  -0.4   0.2   0.5 ]
   [ 0.4   0.1  -0.3  -0.2 ]]]
```
🔹 Each word index is mapped to a **dense vector of size 4** (randomly initialized and learned during training).

---

## 🎯 **Why Use Word Embeddings?**
✅ **Captures word relationships** → Words with similar meanings get similar vector representations.  
✅ **Reduces dimensionality** → Instead of a one-hot vector (huge size), each word gets a fixed-size vector.  
✅ **Improves NLP model performance** → Helps LSTMs/RNNs learn context better.  
