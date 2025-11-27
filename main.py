from datasets.mnist_dataloader import get_mnist_dataloader


def main():
    print("Hello from cs-552-generative-ai-final-project!")

    # Load MNIST Dataloader
    mnist_dataloader = get_mnist_dataloader()
    
    # Print Dataset Summary
    print(f"{'-' * 10} DATASET INFO {'-' * 10}\n")
    print(f"Dataset length: {len(mnist_dataloader.dataset)}")
    print(f"X (Image) shape: {mnist_dataloader.dataset[0][0].shape}")
    print(f"{'-' * 34}")


if __name__ == "__main__":
    main()
