# Import necessary libraries
import tqdm          # For displaying progress bars during training loops
import wandb         # For logging experiment metrics and artifacts (Weights & Biases)
import torch         # PyTorch library for tensor computations and neural networks
import dataset       # Custom module likely containing the dataset class (Wiki)
import evaluate      # Custom module likely containing evaluation functions (topk)
import datetime      # For getting the current timestamp to name runs and checkpoints
import model         # Custom module likely containing the CBOW model definition
import os            # Add os import


# Basic setup and configuration
# Set a fixed seed for reproducibility of random operations (like weight initialization)
torch.manual_seed(42)
# Determine the device to run on: use CUDA GPU if available, otherwise use CPU
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Get the current timestamp as a string for unique naming
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


# Load and prepare the dataset
# Instantiate the dataset object from the custom 'dataset' module
ds = dataset.Wiki()
# Create a DataLoader to handle batching and shuffling of the dataset
# `batch_size=256` means 256 context/target pairs will be processed in each iteration
dl = torch.utils.data.DataLoader(dataset=ds, batch_size=256)


# Initialize the model, optimizer, and loss function
# Define arguments for the CBOW model: vocabulary size and embedding dimension
args = (len(ds.int_to_vocab), 128) # (vocab_size, embedding_dim)
# Instantiate the CBOW model from the custom 'model' module
mFoo = model.CBOW(*args)
# Print the total number of trainable parameters in the model
print('mFoo:params', sum(p.numel() for p in mFoo.parameters()))
# Initialize the Adam optimizer to update model weights
# `lr=0.003` sets the learning rate
opFoo = torch.optim.Adam(mFoo.parameters(), lr=0.003)
# Define the loss function: CrossEntropyLoss is common for classification tasks like predicting the target word
criterion = torch.nn.CrossEntropyLoss()


# Initialize Weights & Biases for experiment tracking
# `project` specifies the project name on W&B
# `name` gives this specific run a unique name using the timestamp
wandb.init(project='mlx7-week1-cbow', name=f'{ts}')
# Move the model to the selected device (GPU or CPU)
mFoo.to(dev)


# Training loop
# Loop over a fixed number of epochs (5 passes through the entire dataset)
for epoch in range(5):
    # Wrap the DataLoader with tqdm to show a progress bar for the current epoch
    prgs = tqdm.tqdm(dl, desc=f'Epoch {epoch+1}', leave=False)
    # Iterate over batches of data provided by the DataLoader
    for i, (ipt, trg) in enumerate(prgs):
        # Move the input context words (ipt) and target word (trg) tensors to the selected device
        ipt, trg = ipt.to(dev), trg.to(dev)
        # Reset the gradients of the optimizer before calculating new gradients
        opFoo.zero_grad()
        # Perform the forward pass: get model predictions for the input context words
        out = mFoo(ipt)
        # Calculate the loss between the model's predictions (out) and the actual target words (trg)
        # `trg.squeeze()` removes any unnecessary dimensions from the target tensor
        loss = criterion(out, trg.squeeze())
        # Perform the backward pass: calculate gradients of the loss with respect to model parameters
        loss.backward()
        # Update the model parameters based on the calculated gradients
        opFoo.step()
        # Log the current batch loss to Weights & Biases
        wandb.log({'loss': loss.item()})
        # Evaluate the model using the top-k accuracy metric every 10,000 batches
        if i % 10_000 == 0:
            evaluate.topk(mFoo)

    # --- Checkpointing after each epoch ---
    # Define a unique filename for the checkpoint based on timestamp and epoch number
    checkpoint_name = f'{ts}.{epoch + 1}.cbow.pth'
    # Ensure the checkpoints directory exists
    os.makedirs('./checkpoints', exist_ok=True)
    # Save the model's learned parameters (state dictionary) to a file
    # Note: This assumes a './checkpoints/' directory exists.
    # If not, this line will cause an error. A check/creation step might be needed.
    torch.save(mFoo.state_dict(), f'./checkpoints/{checkpoint_name}')
    # Create a W&B artifact to store the model weights
    artifact = wandb.Artifact('model-weights', type='model')
    # Add the saved checkpoint file to the artifact
    artifact.add_file(f'./checkpoints/{checkpoint_name}')
    # Log the artifact to W&B, associating the checkpoint file with this run
    wandb.log_artifact(artifact)


# Finish the Weights & Biases run after training is complete
wandb.finish()
