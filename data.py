import matplotlib.pyplot as plt  # only needed for plotting
import torch
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting

DATA_PATH = "dtu_mlops/s1_development_environment/exercise_files/final_exercise/corruptmnist_v1"

def corrupt_mnist():
    """Return train and test datasets for corrupt MNIST."""
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{DATA_PATH}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{DATA_PATH}/train_target_{i}.pt"))
    train_images = torch.cat(train_images) #concatenate into one large tensor
    train_target = torch.cat(train_target)

    test_images = torch.load(f"{DATA_PATH}/test_images.pt")
    test_target = torch.load(f"{DATA_PATH}/test_target.pt")

    train_images = train_images.unsqueeze(1).float() # Grayscale, so need to add a channel to fit into format (batch_size, channels, height, width)
    test_images = test_images.unsqueeze(1).float()

    train_target = train_target.long() # Integer labels of type torch.int64 (long)
    test_target = test_target.long()

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set


def show_image_and_target(images,target):
    """Plot images and their labels in a grid."""
    
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray") # squeeze to remove channel
        ax.set_title(f"True Label: {label.item()}")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    train_set, test_set = corrupt_mnist()
    print(f"Size of training set: {len(train_set)}")
    print(f"Size of test set: {len(test_set)}")
    print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
    print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")
    show_image_and_target(train_set.tensors[0][:25], train_set.tensors[1][:25]) # first 25 train images and labels 

