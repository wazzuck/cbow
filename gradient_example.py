import torch

# -- 1. Setup: Parameters and Input --

# Create a learnable parameter 'w' (weight). Initialize it to 3.0.
# 'requires_grad=True' tells PyTorch: "Track operations on 'w' and be ready 
# to calculate gradients with respect to it."
w = torch.tensor([3.0], requires_grad=True) 

# Create another learnable parameter 'b' (bias). Initialize it to 1.0.
b = torch.tensor([1.0], requires_grad=True)

# Create some input data 'x'. This doesn't need gradients.
x = torch.tensor([2.0]) 

# Define the target value we want our model to eventually output for input 'x'.
target = torch.tensor([10.0])

print(f"Initial weight w: {w.item()}")
print(f"Initial bias b:   {b.item()}")
print(f"Input x:          {x.item()}")
print(f"Target value:     {target.item()}")


# -- 2. Forward Pass: Calculate Output and Loss --

# Perform a simple calculation (our "model"): y = w * x + b
y = w * x + b 
print(f"Calculated output y = w*x + b: {y.item()}")

# Calculate the loss: How far is our output 'y' from the 'target'?
# We'll use Mean Squared Error: (y - target)^2
loss = torch.mean((y - target)**2) 
print(f"Calculated Loss: {loss.item():.4f}")

# At this point, the gradients for w and b are not calculated yet.
print(f"Gradients BEFORE backward():")
print(f"  w.grad: {w.grad}")
print(f"  b.grad: {b.grad}")


# -- 3. Backward Pass: Calculate Gradients --

# This is the core step!
# PyTorch traces back the operations that created 'loss' (y=w*x+b, then loss=(y-target)^2)
# and calculates the gradient of 'loss' with respect to EVERY tensor involved 
# that had 'requires_grad=True' (which are 'w' and 'b').
print("--- Calling loss.backward() ---")
loss.backward() 

# -- 4. Inspect Gradients --

# Now, the .grad attribute of 'w' and 'b' will be populated.
print(f"Gradients AFTER backward():")
print(f"  w.grad: {w.grad.item():.4f}  <-- d(loss)/dw")
print(f"  b.grad: {b.grad.item():.4f}  <-- d(loss)/db")

# -- 5. Interpretation --

print("Interpretation:")
print(f" - The gradient w.grad ({w.grad.item():.4f}) tells us:")
print(f"   If we slightly INCREASE 'w', the loss is expected to INCREASE.") 
print(f"   (Because the gradient is positive)")
print(f" - The gradient b.grad ({b.grad.item():.4f}) tells us:")
print(f"   If we slightly INCREASE 'b', the loss is expected to INCREASE.")
print(f"   (Because the gradient is positive)")

# -- How an Optimizer Would Use This --
# An optimizer (like SGD or Adam) would look at these positive gradients and decide 
# to DECREASE 'w' and 'b' slightly in the next step to try and reduce the loss.
# For example (simplified SGD):
# learning_rate = 0.1
# w_new = w - learning_rate * w.grad 
# b_new = b - learning_rate * b.grad
# (Note: We don't do the update here, just showing the principle) 