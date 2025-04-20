# Import the PyTorch library
import torch


# Define the Continuous Bag-of-Words (CBOW) model
class CBOW(torch.nn.Module):
    """ A simple CBOW model for word embeddings.

    Predicts a target word based on the average of the embeddings
    of its surrounding context words.
    """
    def __init__(self, voc: int, emb: int):
        """ Initialize the CBOW model layers.

        Args:
            voc (int): The size of the vocabulary (number of unique words).
            emb (int): The desired dimensionality of the word embeddings.
        """
        # Call the parent class (torch.nn.Module) constructor
        super().__init__()
        # Embedding layer: maps word indices (integers) to dense vectors (embeddings).
        # `num_embeddings=voc`: size of the dictionary of embeddings (vocabulary size).
        # `embedding_dim=emb`: the size of each embedding vector.
        self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        # Linear layer: projects the averaged context embedding to the vocabulary size.
        # This produces scores for each word in the vocabulary, predicting the target word.
        # `in_features=emb`: size of the input (the averaged embedding vector).
        # `out_features=voc`: size of the output (scores for each word in the vocabulary).
        # `bias=False`: Typically, the output layer in word embedding models doesn't use a bias.
        self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        """ Defines the forward pass of the CBOW model.

        Args:
            inpt (torch.Tensor): A tensor containing the indices of the context words.
                                 Shape: (batch_size, context_window_size)

        Returns:
            torch.Tensor: Output scores for each word in the vocabulary.
                          Shape: (batch_size, vocab_size)
        """
        # 1. Get embeddings for the input context word indices.
        # Input `inpt` shape: (batch_size, context_window_size)
        # Output `emb` shape: (batch_size, context_window_size, embedding_dim)
        emb = self.emb(inpt)
        # 2. Average the embeddings across the context window dimension (dim=1).
        # This creates a single embedding vector representing the average context.
        # Input `emb` shape: (batch_size, context_window_size, embedding_dim)
        # Output `emb` shape: (batch_size, embedding_dim)
        emb = emb.mean(dim=1)
        # 3. Pass the averaged context embedding through the linear layer.
        # Input `emb` shape: (batch_size, embedding_dim)
        # Output `out` shape: (batch_size, vocab_size)
        out = self.ffw(emb)
        # Return the final output scores
        return out


# Define a simple Regressor model
class Regressor(torch.nn.Module):
    """ A simple feed-forward neural network for regression.

    Takes a fixed-size input vector (e.g., an averaged embedding)
    and predicts a single continuous value.
    """
    def __init__(self):
        """ Initialize the Regressor model layers. """
        # Call the parent class (torch.nn.Module) constructor
        super().__init__()
        # Define a sequence of layers: Linear -> ReLU -> Linear -> ReLU ... -> Linear
        self.seq = torch.nn.Sequential(
            # Input layer: 128 features (likely embedding dim) -> 64 features
            torch.nn.Linear(in_features=128, out_features=64),
            # ReLU activation function
            torch.nn.ReLU(),
            # Hidden layer 1: 64 features -> 32 features
            torch.nn.Linear(in_features=64, out_features=32),
            # ReLU activation function
            torch.nn.ReLU(),
            # Hidden layer 2: 32 features -> 16 features
            torch.nn.Linear(in_features=32, out_features=16),
            # ReLU activation function
            torch.nn.ReLU(),
            # Output layer: 16 features -> 1 feature (the regression target)
            torch.nn.Linear(in_features=16, out_features=1),
        )

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        """ Defines the forward pass of the Regressor model.

        Args:
            inpt (torch.Tensor): The input tensor (e.g., averaged embedding).
                                Shape: (batch_size, input_features) (e.g., (batch_size, 128))

        Returns:
            torch.Tensor: The predicted continuous value(s).
                          Shape: (batch_size, 1)
        """
        # Pass the input through the sequential layers
        out = self.seq(inpt)
        # Return the final output
        return out


# Example usage block (runs only when the script is executed directly)
if __name__ == '__main__':
    # Instantiate a small CBOW model for demonstration
    # vocab_size=128, embedding_dim=8
    model = CBOW(128, 8)
    # Print the model architecture
    print('CBOW:', model)
    # Define a loss function (CrossEntropyLoss expects raw scores from the model)
    criterion = torch.nn.CrossEntropyLoss()
    # Create dummy input context words (batch of 3, context window 5, random indices 0-127)
    inpt = torch.randint(0, 128, (3, 5)) # (batch_size, seq_len)
    # Create dummy target words (batch of 3, random indices 0-127)
    trgt = torch.randint(0, 128, (3,))   # (batch_size)
    # Pass the dummy input through the model
    out = model(inpt)
    # Calculate the loss between the model output and the target
    loss = criterion(out, trgt)
    # Print the loss. For an untrained model with random weights,
    # the expected loss is roughly -ln(1/vocab_size).
    # Here, -ln(1/128) = ln(128) which is approximately 4.85.
    print(loss) # Should be around 4.85
