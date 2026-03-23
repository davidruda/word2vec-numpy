
import argparse

import numpy as np

from text8 import Text8

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--max_words", default=1_000_000, type=int, help="Maximum number of words to load from the dataset.")
parser.add_argument("--window_size", default=2, type=int, help="Window size to use.")
parser.add_argument("--learning_rate", default=0.025, type=float, help="Learning rate.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--embedding_dim", default=60, type=int, help="Word embedding dimension.")
parser.add_argument("--negative_samples", default=5, type=int, help="Number of negative samples to use.")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1 + np.exp(-x))


class Word2Vec:
    def __init__(self, args: argparse.Namespace, vocab_size: int):
        # lower_bound = -0.5 / args.embedding_dim
        # upper_bound = 0.5 / args.embedding_dim
        lower_bound = -0.6
        upper_bound = 0.6
        self._W1 = np.random.uniform(lower_bound, upper_bound, size=(vocab_size, args.embedding_dim))
        self._W2 = np.random.uniform(lower_bound, upper_bound, size=(args.embedding_dim, vocab_size))


    def train_step(self, inputs: np.ndarray, targets: np.ndarray, negative_samples: np.ndarray) -> float:
        """
        `inputs` shape: (batch_size,) - word ids of the center words  
        `targets` shape: (batch_size,) - word ids of the context words  
        `negative_samples` shape: (batch_size, num_negative_samples) - word ids of the negative samples
        """

        batch_size = inputs.shape[0]

        # hidden shape: (batch_size, embedding_dim)
        hidden = self._W1[inputs] # use lookup instead of multiplying one-hot vectors
        hidden = hidden[:, np.newaxis, :] # reshape to (batch_size, 1, embedding_dim)

        # shape: (batch_size, embedding_dim, 1)
        W2_pos = self._W2[:, targets, np.newaxis].transpose(1, 0, 2)
        # shape: (batch_size, 1, 1)
        logits_pos = hidden @ W2_pos

        # shape: (batch_size, embedding_dim, num_negative_samples)
        W2_neg = self._W2[:, negative_samples].transpose(1, 0, 2)
        # shape: (batch_size, 1, num_negative_samples)
        logits_neg = hidden @ W2_neg

        probs_pos = sigmoid(logits_pos)
        probs_neg = sigmoid(logits_neg)

        loss = self.compute_loss(probs_pos.squeeze(), probs_neg.squeeze())

        # Compute all necessary gradients for backpropagation
        d_logits_pos = probs_pos - 1
        d_logits_neg = probs_neg # - 0

        # shape: (batch_size, embedding_dim, 1)
        d_W2_pos = hidden * d_logits_pos
        # shape: (batch_size, embedding_dim, num_negative_samples)
        d_W2_neg = hidden.transpose(0, 2, 1) * d_logits_neg

        d_hidden_pos = d_logits_pos @ W2_pos.transpose(0, 2, 1)
        d_hidden_neg = d_logits_neg @ W2_neg.transpose(0, 2, 1)
        d_hidden = d_hidden_pos + d_hidden_neg

        d_W1 = d_hidden

        # We must use `np.add.at` to correctly accumulate gradients for repeated indices in `targets` and `negative_samples`
        update_W2_pos = -args.learning_rate * d_W2_pos.squeeze(axis=1) / batch_size
        np.add.at(self._W2.T, targets, update_W2_pos)

        # Reshape negative_samples to shape (batch_size * num_negative_samples,)
        negative_samples_flat = negative_samples.reshape(-1)
        # Reshape d_W2_neg to (batch_size * num_negative_samples, embedding_dim)
        d_W2_neg_flat = d_W2_neg.transpose(0, 2, 1).reshape(-1, args.embedding_dim)

        update_W2_neg = -args.learning_rate * d_W2_neg_flat / batch_size
        np.add.at(self._W2.T, negative_samples_flat, update_W2_neg)

        update_W1 = -args.learning_rate * d_W1.squeeze(axis=1) / batch_size
        np.add.at(self._W1, inputs, update_W1)

        return loss


    def compute_loss(self, positive_probs: np.ndarray, negative_probs: np.ndarray) -> float:
        # Add 1e-9 for numerical stability to avoid log(0)
        loss = np.mean(-np.log(positive_probs + 1e-9) - np.sum(np.log(1 - negative_probs + 1e-9), axis=1), axis=0)
        return loss


    def train(self, words, prepare_batch_fn):
        for epoch in range(args.epochs):
                total_epoch_loss = 0
                running_loss = 0
                batch_count = 0

                batch_generator = prepare_batch_fn(words)

                for i, batch in enumerate(batch_generator):
                    loss = self.train_step(batch["inputs"], batch["targets"], batch["negative_samples"])

                    total_epoch_loss += loss
                    running_loss += loss
                    batch_count += 1

                    if (i + 1) % 100 == 0:
                        print(f"Epoch {epoch + 1} | Batch {i + 1:,} | Loss (Last 100): {running_loss / 100:.4f}")
                        running_loss = 0

                # Final Epoch Summary
                print(f"{batch_count:,} batches processed.")
                print(f"==> Epoch {epoch + 1} Complete. Average Loss: {total_epoch_loss / batch_count:.4f}")


    def get_similar(self, word_idx: int, top_k: int = 5):
        word_vector = self._W1[word_idx]

        # compute cosine similarity between our world_vector and all other vectors
        # cosine similarity = (A * B) / (||A|| * ||B||)
        norms = np.linalg.norm(self._W1, axis=1) * np.linalg.norm(word_vector)
        similarity = self._W1 @ word_vector / (norms + 1e-9)

        # Get indices of the highest similarities
        closest_indices = np.argsort(similarity)[::-1]
        # Skip the first one because it's the word itself
        closest_indices = closest_indices[1: top_k + 1]

        return closest_indices


def main(args: argparse.Namespace):

    np.random.seed(args.seed)

    text8 = Text8(max_words=args.max_words)
    words = text8.words

    total_words = len(text8.words)
    frequencies = np.array(list(text8.frequencies.values()))
    vocab_size = len(frequencies)

    # Subsample Frequent Words
    z = frequencies / total_words
    # probability of keeping the word
    sampling_rate = (np.sqrt(z /0.001) + 1) * (0.001 / z)


    frequencies_adjusted = frequencies ** (3 / 4)
    negative_sampling_probs = frequencies_adjusted / np.sum(frequencies_adjusted)

    def prepare_data(words: list[str]):
        for idx, word in enumerate(words):
            center = text8.word_to_id[word]

            # Subsample very frequent words
            if np.random.rand() > sampling_rate[center]:
                continue

            for before in text8.tokenize(words[max(0, idx - args.window_size): idx]):
                yield center, before
            for after in text8.tokenize(words[idx + 1: min(idx + args.window_size + 1, len(words))]):
                yield center, after


    def prepare_negative_samples(batch_size: int, num_negative_samples: int, vocab_size: int) -> np.ndarray:
        return np.random.choice(
            vocab_size, p=negative_sampling_probs, size=(batch_size, num_negative_samples), replace=True
        ).astype(int)


    def prepare_batch(words: list[str]):
        data_generator = prepare_data(words)
        inputs, targets = [], []

        for input, target in data_generator:
            inputs.append(input)
            targets.append(target)

            if len(inputs) == args.batch_size:
                negative_samples = prepare_negative_samples(len(inputs), args.negative_samples, vocab_size)

                yield {
                    "inputs": np.array(inputs, dtype=int),
                    "targets": np.array(targets, dtype=int),
                    "negative_samples": negative_samples
                }

                inputs, targets = [], []

        if len(inputs) > 0:
            negative_samples = prepare_negative_samples(len(inputs), args.negative_samples, vocab_size)

            yield {
                "inputs": np.array(inputs, dtype=int),
                "targets": np.array(targets, dtype=int),
                "negative_samples": negative_samples
            }

    word2vec = Word2Vec(args, vocab_size)

    # Train the model
    word2vec.train(words, prepare_batch)

    print(f"First 100 words of text8 dataset:\n{words[:100]}")
    print(f"Total number of words in text8 dataset: {len(words):,}")

    print(f"Vocabulary size: {len(text8.frequencies)}")
    print(f"Most common words: {text8.frequencies.most_common(10)}")

    print(f"Frequency of 'man': {text8.frequencies["man"]}")
    similar_indices = word2vec.get_similar(text8.word_to_id["man"], top_k=5)
    print(f"Similar words to 'man': {list(text8.detokenize(list(similar_indices)))}")


    print(f"Frequency of 'woman': {text8.frequencies["woman"]}")
    similar_indices = word2vec.get_similar(text8.word_to_id["woman"], top_k=5)
    print(f"Similar words to 'woman': {list(text8.detokenize(list(similar_indices)))}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
