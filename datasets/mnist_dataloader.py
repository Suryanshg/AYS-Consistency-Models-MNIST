from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# TODO: Check if this is sufficient
# Top-Level Transforms pipeline
TRANSFORMS_PIPELINE = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]
)

# Global MNIST Dataset
MNIST_DATASET = MNIST(
    "./data",
    train = True,
    download = True,
    transform = TRANSFORMS_PIPELINE
)


# ┌───────────────────────────────────────────────┐
# │                   DATALOADERS                 │
# └───────────────────────────────────────────────┘
def get_mnist_dataloader(batch_size: int = 128) -> DataLoader:
    """_summary_

    Args:
        batch_size (int, optional): _description_. Defaults to 128.

    Returns:
        DataLoader: _description_
    """
    return DataLoader(MNIST_DATASET, shuffle = True, batch_size = batch_size)



# ┌───────────────────────────────────────────────┐
# │                   DRIVER CODE                 │
# └───────────────────────────────────────────────┘
if __name__ == '__main__':
    dataloader = get_mnist_dataloader()
    print(f"{'-' * 10} DATASET INFO {'-' * 10}\n")
    print(f"Dataset length: {len(dataloader.dataset)}")
    print(f"X (Image) shape: {dataloader.dataset[0][0].shape}")
    print(f"{'-' * 34}")