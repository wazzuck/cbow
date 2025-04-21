import torch
import torch.nn as nn

# 1. Create a simple Linear layer
# This layer takes an input vector of size 3 and outputs a vector of size 2
# It will automatically create learnable weights and biases
linear_layer = nn.Linear(in_features=3, out_features=2) 

# 2. Look at the randomly initialized weights
# Weights connect each input feature to each output feature.
# Shape: (output_features, input_features) -> (2, 3)
print("Initial Weights:")
print(linear_layer.weight) 
# tensor([[-0.1596,  0.5114, -0.1104],  # Weights for the first output neuron
#         [-0.4876, -0.0852, -0.3795]], # Weights for the second output neuron
#        requires_grad=True)  # These are learnable parameters

# 3. Look at the randomly initialized biases
# There is one bias term for each output feature.
# Shape: (output_features,) -> (2,)
print("\nInitial Biases:")
print(linear_layer.bias)
# tensor([ 0.1169, -0.2730], requires_grad=True) # Learnable biases

# 4. Example Input
input_data = torch.tensor([1.0, 2.0, 3.0]) # A vector of size 3

# 5. Apply the layer (performs: output = input * weight^T + bias)
output = linear_layer(input_data)
print("\nOutput of the layer:")
print(output)
# tensor([0.6507, -1.8061], grad_fn=<AddBackward0>) # Result after applying weights and biases
