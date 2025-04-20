# Import necessary libraries
import torch  # PyTorch library
import model  # Custom module containing model definitions (CBOW, Regressor)


# Load the pre-trained CBOW model
# Instantiate a CBOW model with the same architecture as the trained one
# Arguments: vocab_size=63642, embedding_dim=128 (These should match the trained model)
cbow = model.CBOW(63642, 128)
# Load the saved weights (state dictionary) from a checkpoint file
# IMPORTANT: The checkpoint path './checkpoints/2025_04_17__11_04_09.5.cbow.pth' is hardcoded.
# This will need to be updated if a different checkpoint is used.
cbow.load_state_dict(torch.load('./checkpoints/2025_04_17__11_04_09.5.cbow.pth'))
# Set the CBOW model to evaluation mode. This disables dropout and batch normalization updates,
# which is important when using the model for inference or feature extraction.
cbow.eval()


# Initialize the Regressor model and its optimizer
# Instantiate the Regressor model (likely defined in the 'model' module)
mReg = model.Regressor()
# Initialize the Adam optimizer for the Regressor model's parameters
opFoo = torch.optim.Adam(mReg.parameters(), lr=0.005)


# Dummy training loop for the Regressor
# This loop runs 100 times using the same hardcoded input and target.
# This is likely a placeholder or a very basic example, not real training.
for i in range(100):
    # Define a hardcoded target score tensor
    trg = torch.tensor([[125.]])               # Target score value
    # Define a hardcoded input sequence of token IDs (representing a title, perhaps?)
    ipt = torch.tensor([[45, 27, 45367, 456]]) # Input token IDs

    # --- Feature Extraction using CBOW ---
    # Get the embeddings for the input token IDs from the pre-trained CBOW model's embedding layer
    # `cbow.emb(ipt)` retrieves the embedding vectors for the sequence [45, 27, 45367, 456]
    # `.mean(dim=1)` calculates the average embedding across the sequence dimension,
    # effectively creating a single fixed-size vector representation for the input sequence.
    emb = cbow.emb(ipt).mean(dim=1)

    # --- Regressor Training Step ---
    # Pass the averaged embedding vector through the Regressor model to get a prediction
    out = mReg(emb)
    # Calculate the L1 loss (Mean Absolute Error) between the prediction (out) and the target (trg)
    loss = torch.nn.functional.l1_loss(out, trg)
    # Calculate gradients of the loss with respect to the Regressor's parameters
    loss.backward()
    # Update the Regressor's parameters using the optimizer
    opFoo.step()
    # Reset the gradients for the next iteration
    opFoo.zero_grad()
    # Print the loss value for this iteration
    print(loss.item())
