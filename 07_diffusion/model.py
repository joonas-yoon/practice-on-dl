import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np


def forward_diffusion(x: np.ndarray, n_trajectory: int, noise_std: float = 0.1) -> list[np.ndarray]:
    """
    Perturb the data over trajectories by adding Gaussian noise.

    Args:
        x: Original data
        n_trajectory: Number of trajectory for the diffusion
        noise_std: Standard deviation of Gaussian noise

    Returns:
        List of noisy data at each timestep
    """
    noise = torch.randn_like(x) * noise_std
    diffusion_data = [x]
    for t in range(n_trajectory):
        x = x + noise
        diffusion_data.append(x)
    return diffusion_data


class ReverseDiffusionNet(nn.Module):
    """
    Define the reverse process network (Score-based model)
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_trajectory: int):
        super(ReverseDiffusionNet, self).__init__()
        self.t_weights = nn.ParameterList()
        for _ in range(n_trajectory):
            layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
            self.t_weights.append(layer)

    def forward(self, x: np.ndarray, t: int):
        # Flatten the input from (batch_size, 28, 28) to (batch_size, 784)
        x = x.view(x.size(0), -1)
        return self.t_weights[t](x)


def score_matching_loss(predicted_score, true_data, noisy_data, noise_std):
    """
    Calculate the score matching loss.

    Args:
        predicted_score: The predicted gradient of the log density
        true_data: The original clean data
        noisy_data: The noisy data at the current timestep
        noise_std: Standard deviation of Gaussian noise added during diffusion

    Returns:
        Score matching loss
    """
    noise = noisy_data - true_data
    score = noise / (noise_std ** 2)
    return torch.mean((predicted_score - score) ** 2)


def generate_image_from_noise(model, n_trajectory=10, noise_std=0.1):
    """
    Generate an image starting from pure noise using the trained model.

    Args:
        model: The trained reverse diffusion model
        n_trajectory: Number of n_trajectory to reverse the diffusion
        noise_std: Standard deviation of the initial noise

    Returns:
        Generated image after reversing the diffusion process
    """
    # Step 1: Start with a random noise image (batch_size, 1, 28, 28)
    noise = torch.randn(batch_size, 1, 28, 28).to(device) * noise_std

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Step 2: Iteratively denoise the image by reversing the diffusion process
        for t in reversed(range(n_trajectory)):
            # Flatten the noise to match the model input shape (batch_size, 784)
            noise_flat = noise.view(noise.size(0), -1)
            # Predict the denoised image
            noise_flat = model(noise_flat, t=t)
            # Reshape back to image format (batch_size, 1, 28, 28)
            noise = noise_flat.view(noise.size(0), 1, 28, 28)

    return noise


def plot_generated_image(image: np.ndarray, output_file: str):
    """
    Plot the generated image.

    Args:
        image: Generated image tensor of shape (batch_size, 1, 28, 28)
    """
    size = int(image.size(0) ** .5)
    fig, axes = plt.subplots(size, size, figsize=(6, 6))
    axes = axes.flatten()

    # Plot noisy images
    for i in range(size * size):
        data = image[i].view(28, 28).cpu().numpy()
        axes[i].imshow(data, cmap='gray')
        axes[i].axis('off')

    plt.tight_layout(pad=1.0, rect=(0, 0, 1, 0.95))
    fig.suptitle('Generated images from noise')
    fig.savefig(output_file)
    plt.close()


def plot_train_losses(train_losses: list[float], filename: str = 'training_loss_plot.png'):
    """
    Plot the training losses and save the plot to a file.

    Args:
        train_losses: List of loss values recorded during training
        filename: The name of the file to save the plot to
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    y_limit_min = np.min(train_losses)
    y_limit_min -= np.quantile(train_losses, 0.1) - y_limit_min
    y_limit_max = np.quantile(train_losses, 0.9)
    plt.ylim((y_limit_min, y_limit_max))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def train(model: nn.Module,
          train_loader: DataLoader,
          optimizer: optim.Optimizer,
          scheduler: optim.lr_scheduler.LRScheduler,
          n_trajectory=10,
          noise_std=0.1,
          epochs=10):

    train_losses = []

    progress = tqdm(total=epochs * len(train_loader))
    step = 0

    for epoch in range(epochs):
        # train loop
        model.train()
        train_batch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).view(batch_x.size(0), -1)
            batch_y = batch_y.to(device).view(batch_y.size(0), -1)
            optimizer.zero_grad()

            # Forward diffusion process
            noisy_data_list = forward_diffusion(
                batch_x, n_trajectory, noise_std)

            # Reverse process: predict score function for each timestep
            total_loss = 0
            for t in range(n_trajectory):
                noisy_data = noisy_data_list[t]
                predicted_score = model(noisy_data, t=t)

                loss = score_matching_loss(
                    predicted_score, batch_x, noisy_data, noise_std)
                total_loss += loss

            total_loss.backward()
            optimizer.step()
            train_batch_loss += total_loss.item()

            # display progress
            step += 1
            progress.set_description(f"Epochs: {step / len(train_loader):.2f}, "
                                     f"Loss: {total_loss.item():.4f}, "
                                     f"LR: {scheduler.get_last_lr()[-1]:.4f}")
            progress.update(1)

        train_losses.append(train_batch_loss / len(train_loader))
        plot_train_losses(train_losses)

        if epoch % 20 == 0:
            # valid loop
            model.eval()
            with torch.no_grad():
                # Generate and plot the image
                generated_image = generate_image_from_noise(
                    model, n_trajectory=n_trajectory, noise_std=noise_std)
                plot_generated_image(
                    generated_image[:16], f"generated_{epoch}.png")

        # end of epoch
        scheduler.step()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


device = get_device()

# Define hyperparameters
input_dim = 28 * 28  # for example, MNIST data flattened
hidden_dim = 512
n_trajectory = 20
noise_std = 0.1
epochs = 2000
batch_size = 256
learning_rate = 0.001

# Instantiate model and optimizer
model = ReverseDiffusionNet(input_dim, hidden_dim, n_trajectory).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.95)

# Assume data_loader is predefined for your dataset (e.g., MNIST or CIFAR-10)
train_loader = DataLoader(
    datasets.MNIST('./mnist', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

test_loader = DataLoader(
    datasets.MNIST('./mnist', train=False, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)


# Train the model
train(model, train_loader, optimizer, scheduler,
      n_trajectory=n_trajectory, noise_std=noise_std, epochs=epochs)


def plot_noise_images(original: np.ndarray,
                      noisy_list: list[np.ndarray],
                      reconstructed: np.ndarray,
                      n_trajectory: int,
                      output_file: str):
    """
    Plot the original, noisy (at different trajectory), and reconstructed images.

    Args:
        original: The original image
        noisy_list: A list of noisy images at different trajectory
        reconstructed: The reconstructed image by the model
        n_trajectory: Number of trajectory in the diffusion process
    """
    fig, axes = plt.subplots(1, n_trajectory + 2, figsize=(15, 2))

    # Plot original image
    axes[0].imshow(original.view(28, 28).cpu().numpy(), cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Plot noisy images
    for i in range(n_trajectory):
        data = noisy_list[i].view(28, 28).cpu().numpy()
        axes[i + 1].imshow(data, cmap='gray')
        axes[i + 1].set_title(f"{i+1}")
        axes[i + 1].axis('off')

    # Plot reconstructed image
    axes[-1].imshow(reconstructed.view(28, 28).cpu().numpy(), cmap='gray')
    axes[-1].set_title("Reconstructed")
    axes[-1].axis('off')

    fig.savefig(output_file)
    plt.close()


# After training, visualize the model's performance on a few examples
model.eval()
with torch.no_grad():
    for batch_data, _ in test_loader:
        batch_data = batch_data.to(device)

        # Take a single example from the batch for visualization
        original_image = batch_data[0]

        # Perform forward diffusion
        noisy_data_list = forward_diffusion(
            original_image.unsqueeze(0), n_trajectory, noise_std)

        # Use the model to reconstruct the image from the noisy version
        reconstructed = noisy_data_list[-1]  # Start from the most noisy data
        for t in reversed(range(n_trajectory)):
            reconstructed = model(reconstructed, t=t)

        # Visualize original, noisy, and reconstructed images
        plot_noise_images(original_image, noisy_data_list,
                          reconstructed, n_trajectory, 'reconstruct.png')

        # Only plot for the first batch
        break
