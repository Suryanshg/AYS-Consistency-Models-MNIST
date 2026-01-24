# Google Colab DDPM Training Guide

This guide explains how to use the provided files to train a DDPM (Denoising Diffusion Probabilistic Models) on Google Colab with GPU acceleration.

## Files Provided

1. **`colab_ddpm_training.py`** - Standalone Python script with all code in one file
2. **`colab_ddpm_training.ipynb`** - Jupyter notebook version (recommended for Colab)

## Method 1: Using Jupyter Notebook (Recommended)

### Step 1: Upload to Google Colab
1. Go to [Google Colab](https://colab.research.google.com)
2. Click "Upload" tab
3. Upload `colab_ddpm_training.ipynb`
4. Open the notebook

### Step 2: Enable GPU
1. Click `Runtime` → `Change runtime type`
2. Select `GPU` as Hardware accelerator
3. Click `Save`

### Step 3: Run the Notebook
1. Simply run all cells in order (Ctrl+F9 or Runtime → Run all)
2. The notebook will:
   - Install required packages
   - Load MNIST dataset
   - Create and train the DDPM model
   - Visualize the training loss
   - Save the model weights

## Method 2: Using Python Script

### Step 1: Upload to Colab
```python
# In a Colab cell, upload the file:
from google.colab import files
files.upload()
```

### Step 2: Run the Script
```python
%run colab_ddpm_training.py
```

Or in terminal:
```bash
python colab_ddpm_training.py
```

## Configuration Options

You can modify training parameters in the script:

```python
# In the main() function:
trained_model, loss_history = train(
    model,
    dataloader,
    optimizer,
    num_epochs=10,              # Number of training epochs
    num_diffusion_steps=1000,   # Diffusion timesteps
    schedule_type="linear"      # "linear" or "cosine"
)
```

### Batch Size
```python
dataloader = get_mnist_dataloader(batch_size=64)
# Increase for faster training if GPU memory allows
# Typical: 64 (safe), 128 (fast), 256 (very fast but uses more memory)
```

### Learning Rate
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
# Adjust lr if needed: higher = faster but less stable, lower = slower but more stable
```

## Expected Performance

### On Google Colab GPU (T4/A100)
- **Batch size 64**: ~30-50 sec/epoch
- **Batch size 128**: ~20-30 sec/epoch
- **Total training time (10 epochs)**: ~5-8 minutes

### Training Loss
- **Initial loss**: ~0.5-1.0
- **Final loss**: ~0.01-0.05
- **Reduction**: 90-98%

## Output

After training completes:
1. **Model file**: `ddpm_model.pth` - Contains trained weights
2. **Loss plot**: Shows training progression
3. **Console output**: Training statistics

## Loading Trained Model

```python
# Load the model in Colab
model = DDPMUNet().to(DEVICE)
model.load_state_dict(torch.load('ddpm_model.pth'))
model.eval()

print("Model loaded successfully!")
```

## Troubleshooting

### Out of Memory Error
- Reduce batch size: `batch_size=32`
- Reduce number of epochs: `num_epochs=5`
- Reduce feature_map_dim: `feature_map_dim=32`

### Slow Training
- Increase batch size (if memory allows)
- Verify GPU is being used: Check device shows `cuda`
- Check GPU utilization: Monitor in Colab runtime

### Model Not Saving
- Ensure you have write permissions in `/content/`
- Model is automatically saved to `ddpm_model.pth`
- Can also download it: `files.download('ddpm_model.pth')`

## Downloading Results

```python
# In Colab, download the model
from google.colab import files
files.download('ddpm_model.pth')
```

## Next Steps

After training, you can:
1. Generate new images using the trained model
2. Evaluate model quality with FID scores
3. Fine-tune on different datasets
4. Compare with other diffusion models

## Hardware Requirements

- **Minimum**: Any Colab GPU (T4) - 16GB VRAM
- **Recommended**: A100 GPU - 40GB VRAM
- **CPU fallback**: Supported but very slow (~10 min/epoch)

## References

- DDPM Paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Implementation based on standard DDPM architecture
- Tested on PyTorch 2.0+

## Support

If you encounter issues:
1. Check all imports are successful
2. Verify GPU is enabled (Runtime → Change runtime type)
3. Ensure all cells are run in order
4. Check console output for specific error messages
5. Try reducing batch size first

---

**Happy training! 🚀**
