# Function to compute statistics
def compute_statistics(tensor):
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "max": tensor.max().item(),
        "min": tensor.min().item(),
    }

# Forward hook to capture features
def forward_hook(module, input, output):
    stats[current_dataset][module.__class__.__name__ + "_features"] = compute_statistics(output.detach())

# Backward hook to capture gradients
def backward_hook(module, grad_input, grad_output):
    stats[current_dataset][module.__class__.__name__ + "_gradients"] = compute_statistics(grad_output[0].detach())


