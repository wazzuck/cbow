# Import necessary libraries
import torch         # PyTorch library for tensor operations and data utilities
import pickle        # For loading the pre-saved Python objects (corpus, dictionaries)


# Define a custom PyTorch Dataset for the text data
class Wiki(torch.utils.data.Dataset):
    """ PyTorch Dataset for loading the preprocessed Wikipedia corpus.

    It loads the vocabulary mappings and the tokenized corpus created by
    `00_train_tkn.py`. It provides `__getitem__` to return context/target pairs
    suitable for training a CBOW model.
    """
    def __init__(self):
        """ Initialize the dataset by loading preprocessed data from pickle files. """
        # Load the word -> integer ID mapping dictionary
        self.vocab_to_int = pickle.load(open('./tkn_words_to_ids.pkl', 'rb'))
        # Load the integer ID -> word mapping dictionary
        self.int_to_vocab = pickle.load(open('./tkn_ids_to_words.pkl', 'rb'))
        # Load the list of words (the corpus)
        # Note: This might be memory-intensive if the corpus is huge.
        # It might be more efficient to load tokens directly if they were saved.
        self.corpus = pickle.load(open('./corpus.pkl', 'rb'))
        # Convert the list of words (corpus) into a list of integer token IDs
        # This duplicates the tokenization done in 00_train_tkn.py. It might be
        # more efficient if 00_train_tkn.py saved the `tokens` list directly.
        self.tokens = [self.vocab_to_int[word] for word in self.corpus]

    def __len__(self) -> int:
        """ Return the total number of tokens (words) in the dataset. """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """ Get a single training sample (context words and target word).

        Args:
            idx (int): The index of the target word in the `self.tokens` list.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Context words: A tensor containing the token IDs of the 2 words before
                  and the 2 words after the target word (padded with 0s if at edges).
                  Shape: (4,)
                - Target word: A tensor containing the token ID of the target word.
                  Shape: (1,)
        """
        # The target word is the token at the given index `idx`
        ipt = self.tokens[idx]
        # Get the 2 preceding words (context)
        prv = self.tokens[idx-2:idx]
        # Get the 2 succeeding words (context)
        nex = self.tokens[idx+1:idx+3]

        # --- Padding --- handle cases near the beginning or end of the corpus
        # If fewer than 2 preceding words exist (i.e., near the start),
        # pad the beginning of the `prv` list with 0s (the <PAD> token ID).
        if len(prv) < 2:
            prv = [0] * (2 - len(prv)) + prv
        # If fewer than 2 succeeding words exist (i.e., near the end),
        # pad the end of the `nex` list with 0s.
        if len(nex) < 2:
            nex = nex + [0] * (2 - len(nex))

        # Concatenate the previous and next context words
        # Convert the resulting list of 4 context token IDs and the single target token ID
        # into PyTorch tensors.
        return torch.tensor(prv + nex), torch.tensor([ipt])


# Example usage block (runs only when the script is executed directly)
if __name__ == '__main__':
    # Instantiate the dataset
    ds = Wiki()
    # Print the first 15 token IDs from the loaded tokens list
    print(ds.tokens[:15])
    # # Example: Get the sample where the word at index 0 is the target (will have padding)
    print(ds[0])
    # Example: Get the sample where the word at index 5 is the target
    print(ds[5])
    # Create a DataLoader to test batching
    dl = torch.utils.data.DataLoader(dataset=ds, batch_size=3)
    # Get the first batch from the DataLoader
    ex = next(iter(dl))
    # Print the batch (will contain 3 context tensors and 3 target tensors)
    print(ex)