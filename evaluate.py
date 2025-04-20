# Import necessary libraries
import torch   # PyTorch library for tensor operations
import pickle  # For loading the pre-saved vocabulary mapping dictionaries


# Load vocabulary mappings (created by 00_train_tkn.py)
# These are loaded when the module is imported, making them globally accessible within this file.
vocab_to_int = pickle.load(open('./tkn_words_to_ids.pkl', 'rb')) # Word -> ID mapping
int_to_vocab = pickle.load(open('./tkn_ids_to_words.pkl', 'rb')) # ID -> Word mapping


# Define a function to find and print the top-k most similar words to a target word
def topk(mFoo: torch.nn.Module):
    """ Calculates and prints the top-k words most similar to 'computer'.

    Similarity is measured by cosine similarity between word embeddings.

    Args:
        mFoo (torch.nn.Module): The trained CBOW model instance which contains
                                  the learned embedding layer (`mFoo.emb`).
    """

    # Get the integer ID for the target word 'computer' from the loaded mapping
    idx = vocab_to_int['computer']
    # Get the embedding vector for 'computer' directly from the model's embedding layer weight matrix.
    # `mFoo.emb.weight` is the tensor containing all word embeddings.
    # `[idx]` selects the row corresponding to the word 'computer'.
    # `.detach()` creates a new tensor that doesn't require gradients, important for inference.
    vec = mFoo.emb.weight[idx].detach()

    # Disable gradient calculations for the following operations, as we are only evaluating.
    with torch.no_grad():

        # --- Calculate Cosine Similarities --- #
        # 1. Normalize the target word's embedding vector to have unit L2 norm.
        # `unsqueeze(0)` adds a batch dimension (shape [1, embedding_dim]) needed for normalize.
        vec = torch.nn.functional.normalize(vec.unsqueeze(0), p=2, dim=1)
        # 2. Normalize all embedding vectors in the embedding matrix to have unit L2 norm.
        emb = torch.nn.functional.normalize(mFoo.emb.weight.detach(), p=2, dim=1)
        # 3. Calculate the cosine similarity between the normalized target vector and all normalized
        #    embedding vectors using matrix multiplication.
        # `vec.squeeze()` removes the batch dimension from the target vector (shape [embedding_dim]).
        # Result `sim` is a tensor containing similarity scores for all words in the vocabulary.
        # Shape: (vocab_size,)
        sim = torch.matmul(emb, vec.squeeze())

        # --- Get Top-K Similar Words --- #
        # Find the top 6 similarity scores and their corresponding indices.
        # We ask for 6 because the most similar word will be 'computer' itself.
        top_val, top_idx = torch.topk(sim, 6)

        # Print the results
        print('\nTop 5 words similar to "computer":')
        count = 0
        # Iterate through the top indices and values
        for i, idx_tensor in enumerate(top_idx):
            # Get the integer index from the tensor
            idx_item = idx_tensor.item()
            # Look up the corresponding word using the ID -> word mapping
            word = int_to_vocab[idx_item]
            # Get the similarity score
            sim_score = top_val[i].item()
            # Print the word and its similarity score, formatted to 4 decimal places
            print(f'  {word}: {sim_score:.4f}')
            count += 1
            # Stop after printing the top 5 (excluding the word itself if it appeared first)
            if count == 5:
                break
