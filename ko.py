import numpy as np
import random
import string
import re
from collections import Counter
from itertools import chain
from typing import List, Tuple

def load_data(filename: str) -> str:
    """
    Loads a text file and returns its contents as a string.
    """
    with open(filename, 'r') as f:
        data = f.read()
    return data


def preprocess_data(data: str) -> List[str]:
    """
    Preprocesses the input text by removing special characters and converting
    all characters to lowercase. Returns a list of words.
    """
    data = data.lower()
    data = re.sub(r'[^\w\s]', '', data)
    words = data.split()
    return words


def build_vocab(words: List[str], vocab_size: int) -> Tuple[dict, dict]:
    """
    Builds a vocabulary dictionary and a reverse vocabulary dictionary mapping
    words to indices and vice versa. Only includes the most frequent words up
    to a maximum of `vocab_size`.
    """
    word_counts = Counter(words)
    vocab = dict()
    reverse_vocab = dict()
    for i, (word, count) in enumerate(word_counts.most_common(vocab_size)):
        vocab[word] = i + 1
        reverse_vocab[i + 1] = word
    return vocab, reverse_vocab


def encode_text(words: List[str], vocab: dict) -> List[int]:
    """
    Encodes a list of words using a vocabulary dictionary.
    """
    return [vocab[word] for word in words if word in vocab]


def decode_text(encoded: List[int], reverse_vocab: dict) -> str:
    """
    Decodes a list of encoded words using a reverse vocabulary dictionary.
    """
    return ' '.join([reverse_vocab[code] for code in encoded])

#Now we can define the main transformer model:


class Transformer:
    def __init__(self, vocab_size: int, embedding_dim: int, num_layers: int,
                 hidden_dim: int, num_heads: int, dropout_prob: float,
                 l2_reg: float, learning_rate: float):
        """
        Initializes the transformer model with the specified hyperparameters.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate

        # Initialize embedding layer
        self.embedding_layer = np.random.rand(vocab_size, embedding_dim)

        # Initialize encoding layers
        self.encoding_layers = []
        for i in range(num_layers):
            self.encoding_layers.append(self.build_encoding_layer())

        # Initialize decoding layers
        self.decoding_layers = []
        for i in range(num_layers):
            self.decoding_layers.append(self.build_decoding_layer())

        # Initialize output layer
        self.output_layer = np.random.rand(hidden_dim, vocab_size)

        # Initialize optimizer
        self.optimizer = Adam(learning_rate=learning_rate)

    def build_encoding_layer(self):
        """
        Builds an encoding layer for the transformer model.
        """
        # Initialize self-attention layer
        self_attention_layer = SelfAttention(num_heads=self.num_heads,
                                              hidden_dim=self.hidden_dim)

        # Initialize normalization layer
        normalization_layer = LayerNormalization()

        # Initialize feed-forward layer
        feed_forward_layer = FeedForward(hidden_dim=self.hidden_dim)

        # Initialize dropout layer
        dropout_layer = Dropout(self.dropout_prob)

        # Return encoding layer
        return (self_attention_layer, normalization_layer,
                feed_forward_layer, dropout_layer)

    def build_decoding_layer(self):
        """
        Builds a decoding layer for the transformer model.
        """
        # Initialize self-attention layer
        self_attention_layer = SelfAttention(num_heads=self.num_heads,
                                              hidden_dim=self.hidden_dim)

        # Initialize normalization layer
        normalization_layer = LayerNormalization()

        # Initialize encoder-decoder attention layer
        encoder_decoder_attention_layer = EncoderDecoderAttention(num_heads=self.num_heads,
                                                                   hidden_dim=self.hidden_dim)

        # Initialize feed-forward layer
        feed_forward_layer = FeedForward(hidden_dim=self.hidden_dim)

        # Initialize dropout layer
        dropout_layer = Dropout(self.dropout_prob)

        # Return decoding layer
        return (self_attention_layer, normalization_layer,
                encoder_decoder_attention_layer, feed_forward_layer,
                dropout_layer)

    def train(self, data: List[str], num_epochs: int, batch_size: int):
        """
        Trains the transformer model on the input data for the specified number
        of epochs and batch size.
        """
        # Preprocess data
        words = preprocess_data(data)
        vocab, reverse_vocab = build_vocab(words, self.vocab_size)
        encoded = encode_text(words, vocab)

        # Train model
        for epoch in range(num_epochs):
            # Shuffle training data
            random.shuffle(encoded)

            # Split training data into batches
            num_batches = len(encoded) // batch_size
            batches = np.array_split(encoded[:num_batches * batch_size], num_batches)

            # Train on each batch
            for batch in batches:
                # Initialize gradients
                grad_embedding_layer = np.zeros_like(self.embedding_layer)
                grad_encoding_layers = []
                for i in range(self.num_layers):
                    grad_encoding_layers.append(self.build_encoding_layer())
                grad_decoding_layers = []
                for i in range(self.num_layers):
                    grad_decoding_layers.append(self.build_decoding_layer())
                grad_output_layer = np.zeros_like(self.output_layer)

                # Train on each example in batch
                for example in batch:
                    # Initialize input and target sequences
                    input_seq = example[:-1]
                    target_seq = example[1:]

                    # Encode input sequence
                    encoding = self.encode_sequence(input_seq)

                    # Decode target sequence
                    decoding = self.decode_sequence(encoding, target_seq)

                    # Compute loss and gradients
                    loss, grad_output, grad_decoding, grad_encoding = self.compute_gradients(decoding)

                    # Accumulate gradients
                    grad_embedding_layer[input_seq] += grad_encoding[0]
                    for i in range(self.num_layers):
                        grad_encoding_layers[i][0] += grad_encoding[1][i][0]
                        grad_encoding_layers[i][1] += grad_encoding[1][i][1]
                        grad_encoding_layers[i][2] += grad_encoding[1][i][2]
                        grad_encoding_layers[i][3] += grad_encoding[1][i][3]
                    for i in range(self.num_layers):
                        grad_decoding_layers[i][0] += grad_decoding[0][i][0]
                        grad_decoding_layers[i][1] += grad_decoding[0][i][1]
                        grad_decoding_layers[i][2] += grad_decoding[1][i][0]
                        grad_decoding_layers[i][3] += grad_decoding[1][i][1]
                        grad_decoding_layers[i][4] += grad_decoding[1][i][2]
                    grad_output_layer += grad_output

                # Update parameters using gradients
                self.embedding_layer -= self.optimizer.compute_update(grad_embedding_layer)
                for i in range(self.num_layers):
                    self.encoding_layers[i][0].update_parameters(grad_encoding_layers[i][0])
                    self.encoding_layers[i][1].update_parameters(grad_encoding_layers[i][1])
                    self.encoding_layers[i][2].update_parameters(grad_encoding_layers[i][2])
                    self.encoding_layers[i][3].update_parameters(grad_encoding_layers[i][3])
                    self.decoding_layers[i][0].update_parameters(grad_decoding_layers[i][0])
                    self.decoding_layers[i][1].update_parameters(grad_decoding_layers[i][1])
                    self.decoding_layers[i][2].update_parameters(grad_decoding_layers[i][2])
                    self.decoding_layers[i][3].update_parameters(grad_decoding_layers[i][3])
                    self.decoding_layers[i][4].update_parameters(grad_decoding_layers[i][4])
                self.output_layer -= self.optimizer.compute_update(grad_output_layer)

                # Print loss
                if (batch + 1) % 10 == 0:
                    print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch + 1, batch + 1, loss))

        # Save learned data to file
        with open('pretrain.txt', 'w') as f:
            f.write('{}\n'.format(self.vocab_size))
            f.write('{}\n'.format(self.embedding_dim))
            f.write('{}\n'.format(self.num_layers))
            f.write('{}\n'.format(self.hidden_dim))
            f.write('{}\n'.format(self.num_heads))
            f.write('{}\n'.format(self.dropout_prob))
            f.write('{}\n'.format(self.l2_reg))
            f.write('{}\n'.format(self.learning_rate))
            for word in reverse_vocab.values():
                f.write('{}\n'.format(word))
            for embedding in self.embedding_layer:
                f.write('{}\n'.format(embedding))
            for encoding_layer in self.encoding_layers:
                for layer in encoding_layer:
                    for parameter in layer.parameters:
                        f.write('{}\n'.format(parameter))
            for decoding_layer in self.decoding_layers:
                for layer in decoding_layer:
                    for parameter in layer.parameters:
                        f.write('{}\n'.format(parameter))
            for output in self.output_layer:
                f.write('{}\n'.format(output))

    def encode_sequence(self, sequence: List[int]) -> List[np.ndarray]:
        """
        Encodes a sequence of input tokens using the transformer model.
        """
        # Initialize encoding with embeddings
        encoding = [self.embedding_layer[token] for token in sequence]

        # Apply encoding layers
        for i in range(self.num_layers):
            self_attention_layer, normalization_layer, feed_forward_layer, dropout_layer = self.encoding_layers[i]
            encoding = self_attention_layer.forward(encoding)
            encoding = normalization_layer.forward(encoding)
            encoding = feed_forward_layer.forward(encoding)
            encoding = dropout_layer.forward(encoding)

        return encoding

    def decode_sequence(self, encoding: List[np.ndarray], sequence: List[int]) -> List[np.ndarray]:
        """
        Decodes a sequence of target tokens using the transformer model.
        """
        # Initialize decoding with embeddings
        decoding = [self.embedding_layer[token] for token in sequence[:-1]]

        # Apply decoding layers
        for i in range(self.num_layers):
            self_attention_layer, normalization_layer, encoder_decoder_attention_layer, feed_forward_layer, dropout_layer = self.decoding_layers[i]
            decoding = self_attention_layer.forward(decoding)
            decoding = normalization_layer.forward(decoding)
            decoding = encoder_decoder_attention_layer.forward(decoding, encoding)
            decoding = feed_forward_layer.forward(decoding)
            decoding = dropout_layer.forward(decoding)

        return decoding

    def compute_gradients(self, decoding: List[np.ndarray]) -> Tuple[float, np.ndarray, List[Tuple[np.ndarray]], List[Tuple[List[np.ndarray]]]]:
        """
        Computes the gradients for a decoded target sequence.
        """
        # Compute loss
        logits = np.array([np.dot(decoding[i], self.output_layer) for i in range(len(decoding))])
        loss = np.sum(np.log(np.sum(np.exp(logits), axis=1)) - np.sum(logits[np.arange(len(decoding)), target_seq], axis=1))
        loss += self.l2_reg * (np.sum(self.embedding_layer ** 2) + np.sum(self.output_layer ** 2))
        for i in range(self.num_layers):
            self_attention_layer, normalization_layer, feed_forward_layer, dropout_layer = self.encoding_layers[i]
            loss += self.l2_reg * (np.sum(self_attention_layer.W_q ** 2) + np.sum(self_attention_layer.W_k ** 2) + np.sum(self_attention_layer.W_v ** 2) + np.sum(normalization_layer.gamma ** 2) + np.sum(feed_forward_layer.W_1 ** 2) + np.sum(feed_forward_layer.W_2 ** 2))
            self_attention_layer, normalization_layer, encoder_decoder_attention_layer, feed_forward_layer, dropout_layer = self.decoding_layers[i]
            loss += self.l2_reg * (np.sum(self_attention_layer.W_q ** 2) + np.sum(self_attention_layer.W_k ** 2) + np.sum(self_attention_layer.W_v ** 2) + np.sum(normalization_layer.gamma ** 2) + np.sum(encoder_decoder_attention_layer.W_q ** 2) + np.sum(encoder_decoder_attention_layer.W_k ** 2) + np.sum(encoder_decoder_attention_layer.W_v ** 2) + np.sum(feed_forward_layer.W_1 ** 2) + np.sum(feed_forward_layer.W_2 ** 2))

        # Compute gradients
        grad_logits = np.exp(logits) / np.sum(np.exp(logits), axis=1)[:, None]
        grad_logits[np.arange(len(decoding)), target_seq] -= 1
        grad_output = np.dot(decoding.T, grad_logits)
        grad_decoding = [None] * len(decoding)
        for i in range(len(decoding)):
            grad_decoding[i] = np.dot(grad_logits[i], self.output_layer.T) * (1 - self.dropout_prob)
        grad_encoding = [None] * self.num_layers
        grad_encoding[-1] = self_encoding_layers[-1][0].backward(grad_decoding)
        for i in range(self.num_layers - 2, -1, -1):
            grad_encoding[i] = self_encoding_layers[i][3].backward(grad_encoding[i + 1])
            grad_encoding[i] = self_encoding_layers[i][2].backward(grad_encoding[i])
            grad_encoding[i] = self_encoding_layers[i][1].backward(grad_encoding[i])
            grad_encoding[i] = self_encoding_layers[i][0].backward(grad_encoding[i])
        grad_encoding[0] = self_encoding_layers[0][0].backward(grad_encoding[1])[1]

        return loss, grad_output, grad_decoding, grad_encoding

    def generate_text(self, prompt: str, max_length: int) -> str:
        """
        Generates text from the transformer model given a prompt and a maximum
        output length.
        """
        # Preprocess prompt
        prompt = preprocess_data(prompt)
        encoded_prompt = encode_text(prompt, self.vocab)

        # Initialize encoding with embeddings
        encoding = [self.embedding_layer[token] for token in encoded_prompt]

        # Apply encoding layers
        for i in range(self.num_layers):
            self_attention_layer, normalization_layer, feed_forward_layer, dropout_layer = self.encoding_layers[i]
            encoding = self_attention_layer.forward(encoding)
            encoding = normalization_layer.forward(encoding)
            encoding = feed_forward_layer.forward(encoding)
            encoding = dropout_layer.forward(encoding)

        # Generate text
        decoded_prompt = prompt
        for i in range(max_length):
            # Decode last token
            decoding = self.decode_sequence(encoding, [encoded_prompt[-1]])
            logits = np.dot(decoding[0], self.output_layer)
            proba = np.exp(logits) / np.sum(np.exp(logits))
            token = np.random.choice(len(proba), p=proba)

            # Append token to decoded sequence
            decoded_prompt.append(token)

            # Update encoding with new token
            encoding = self.encode_sequence(decoded_prompt)[-len(encoded_prompt):]

            # Break if generated end token
            if token == self.vocab.index(END_TOKEN):
                break

        # Convert decoded sequence to text
        generated_text = decode_text(decoded_prompt, self.vocab)
        return generated_text
 
