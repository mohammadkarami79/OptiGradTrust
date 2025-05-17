# Alzheimer's MRI Dataset for Federated Learning

This document provides instructions for downloading and using the Alzheimer's MRI dataset with the federated learning framework.

## Dataset Information

The Alzheimer's MRI dataset contains brain MRI scans classified into four categories:
- No Impairment
- Very Mild Impairment
- Mild Impairment
- Moderate Impairment

## Downloading the Dataset

To use the Alzheimer's dataset, you need to download it from Kaggle:

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy)
2. Download the dataset (you may need a Kaggle account)
3. Extract the dataset to your local machine

## Setting Up the Dataset

1. Extract the dataset to the directory specified in the config file (`ALZHEIMER_DATA_DIR`)
2. Ensure the directory structure follows this pattern:
   ```
   ALZHEIMER_DATA_DIR/
   ├── train/
   │   ├── Mild Impairment/
   │   ├── Moderate Impairment/
   │   ├── No Impairment/
   │   └── Very Mild Impairment/
   └── test/
       ├── Mild Impairment/
       ├── Moderate Impairment/
       ├── No Impairment/
       └── Very Mild Impairment/
   ```

## Configuration

To use the Alzheimer's dataset with the federated learning framework, update the following parameters in `config.py`:

```python
# Dataset configuration
DATASET = 'ALZHEIMER'  # Change from 'MNIST' to 'ALZHEIMER'

# Model configuration
MODEL = 'RESNET50'  # Change from 'CNN' to 'RESNET50'

# Alzheimer's dataset configuration
ALZHEIMER_DATA_DIR = './data/alzheimer'  # Update to your actual dataset path
ALZHEIMER_IMG_SIZE = 224  # Size to resize images to
ALZHEIMER_CLASSES = 4  # Number of classes

# ResNet configuration
RESNET_UNFREEZE_LAYERS = 20  # Number of layers to unfreeze from the end
RESNET_PRETRAINED = True  # Whether to use pretrained weights
```

## Common Issues and Solutions

### Pretrained Weight Download Issues

If you encounter network errors when trying to download pretrained weights:

1. **First Attempt:** Try again with a stable internet connection. The weights file size is about 100MB.
2. **If Download Fails:** You can temporarily set `RESNET_PRETRAINED = False` in config.py as a workaround.
3. **Manual Download Option:** You can manually download the weights file and place it in the cache directory:
   - File URL: https://download.pytorch.org/models/resnet50-0676ba61.pth
   - Place in: `~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`
   - Then set `RESNET_PRETRAINED = True` again

**Note:** The download happens only once. After successful download, the weights are cached and will be reused without requiring internet connection.

### Memory Issues

If you encounter memory issues:

1. Reduce `BATCH_SIZE` in config.py
2. Reduce `RESNET_UNFREEZE_LAYERS` to unfreeze fewer layers

## Running the Framework

Once the dataset is properly set up and the configuration is updated, you can run the federated learning system normally:

```
python -m federated_learning.main
```

## Notes

- The ResNet50 model requires significantly more computational resources than the CNN model used for MNIST
- Training on CPUs will be much slower; GPU is recommended for the ResNet model
- You may need to adjust batch sizes and learning rates for optimal performance with this dataset 