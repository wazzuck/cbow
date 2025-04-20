# Import necessary libraries
import collections # Used for counting word frequencies (Counter)
import requests    # Used to download the text dataset from a URL
import pickle      # Used to serialize and save Python objects (like lists and dictionaries) to files


# Download the text8 dataset if it doesn't exist
# text8 is a benchmark dataset containing the first 10^8 characters of English Wikipedia dump.
r = requests.get('https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8')
# Open a file named 'text8' in binary write mode ('wb')
with open('text8', 'wb') as f:
    # Write the downloaded content (bytes) to the file
    f.write(r.content)
# Open the downloaded 'text8' file in text read mode ('r')
with open('text8') as f:
    # Read the entire content of the file into the text8 variable
    text8: str = f.read()


# Define a function to preprocess the raw text
def preprocess(text: str) -> list[str]:
    # Convert the entire text to lowercase
    text = text.lower()
    # Replace punctuation with spaced tokens to treat them as separate words
    # This prevents them from being attached to words (e.g., "word." becomes "word <PERIOD>")
    text = text.replace('.',  ' <PERIOD> ')
    text = text.replace(',',  ' <COMMA> ')
    text = text.replace('"',  ' <QUOTATION_MARK> ')
    text = text.replace(';',  ' <SEMICOLON> ')
    text = text.replace('!',  ' <EXCLAMATION_MARK> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace('(',  ' <LEFT_PAREN> ')
    text = text.replace(')',  ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?',  ' <QUESTION_MARK> ') # Duplicate replacement, likely harmless
    text = text.replace(':',  ' <COLON> ')
    # Split the processed text into a list of words based on spaces
    words = text.split()
    # Count the frequency of each word in the list
    stats = collections.Counter(words)

    # Filter out words that appear 5 times or fewer (remove rare words)
    words = [word for word in words if stats[word] > 5]
    # Return the list of frequent words
    return words


# Apply the preprocessing function to the downloaded text8 data
corpus: list[str] = preprocess(text8)
# Print some information about the resulting corpus
print(type(corpus)) # Expected output: <class 'list'>
print(len(corpus))  # Expected output: ~16,680,599 (number of words after filtering)
print(corpus[:7])   # Print the first 7 words of the processed corpus


# Save the processed corpus (list of words) to a file using pickle
# This allows later scripts to load the corpus without reprocessing the text.
# 'wb' means write in binary mode, which pickle requires.
with open('corpus.pkl', 'wb') as f:
    pickle.dump(corpus, f)


# Define a function to create word-to-integer and integer-to-word mapping dictionaries
def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    # 1. Count Word Frequencies:
    # Count the frequency of each word in the input list (the filtered corpus)
    word_counts = collections.Counter(words)

    # 2. Create Sorted Vocabulary:
    # Create a vocabulary list sorted by word frequency in descending order
    # `key=lambda k: word_counts.get(k)` sorts based on the count of each word `k`
    vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)

    # 3. Create Integer-to-Word Mapping (ID -> Word):
    # Create a dictionary mapping integers (starting from 1) to words in the vocabulary
    # `enumerate(vocab)` provides pairs of (index, word)
    int_to_vocab = {ii+1: word for ii, word in enumerate(vocab)}
    # Add a special token '<PAD>' at index 0, often used for padding sequences
    int_to_vocab[0] = '<PAD>'

    # 4. Create Word-to-Integer Mapping (Word -> ID):
    # Create the reverse mapping: dictionary mapping words to their corresponding integers
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    # 5. Return Both Mappings:
    # Return both mapping dictionaries
    return vocab_to_int, int_to_vocab


# Generate the lookup tables using the processed corpus
words_to_ids, ids_to_words = create_lookup_tables(corpus)
# Convert the entire corpus (list of words) into a list of corresponding integer IDs
tokens = [words_to_ids[word] for word in corpus]
# Print some information about the resulting list of tokens
print(type(tokens)) # Expected output: <class 'list'>
print(len(tokens))  # Expected output: ~16,680,599 (same length as corpus)
print(tokens[:7])   # Print the integer IDs for the first 7 words


# Demonstrate the usage of the created lookup tables
print(ids_to_words[5234])        # Look up word corresponding to ID 5234
print(words_to_ids['the'])       # Look up ID corresponding to the word 'anarchism'
print(words_to_ids['anarchism']) # Look up ID corresponding to the word 'anarchism'
print(words_to_ids['have'])      # Look up ID corresponding to the word 'have'
print(len(words_to_ids))         # Print the total size of the vocabulary (number of unique words + <PAD>)


# Save the word-to-ID and ID-to-word lookup tables to separate files using pickle
# These files will be loaded by other scripts that need these mappings.
with open('tkn_words_to_ids.pkl', 'wb') as f:
    pickle.dump(words_to_ids, f)
with open('tkn_ids_to_words.pkl', 'wb') as f:
    pickle.dump(ids_to_words, f)
