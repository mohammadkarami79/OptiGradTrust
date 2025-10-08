"""
Preprocessing Documentation
============================

Documents the complete preprocessing pipeline for reproducibility.
Addresses Reviewer 1's concern about preprocessing details.
"""

import json
from pathlib import Path
from datetime import datetime
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from federated_learning.config.config import *


def document_preprocessing(output_dir=None):
    """Document preprocessing pipeline details."""
    
    print(f"\n{'📝'*40}")
    print(f"PREPROCESSING PIPELINE DOCUMENTATION")
    print(f"{'📝'*40}\n")
    
    if output_dir is None:
        output_dir = Path('experiments/results/preprocessing_docs')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    documentation = {
        'dataset': DATASET,
        'preprocessing_steps': {
            'MNIST': [
                '1. Load from torchvision.datasets.MNIST',
                '2. Normalize: mean=0.1307, std=0.3081',
                '3. Convert to tensor',
                '4. Resize: 28x28 (grayscale)'
            ],
            'CIFAR10': [
                '1. Load from torchvision.datasets.CIFAR10',
                '2. Normalize: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]',
                '3. Convert to tensor',
                '4. Resize: 32x32 (RGB)'
            ],
            'ALZHEIMER': [
                '1. Load from custom dataset directory',
                '2. Resize: 128x128 (RGB)',
                '3. Normalize: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])',
                '4. Data augmentation: Random horizontal flip (50%)',
                '5. Convert to tensor'
            ]
        },
        'data_split': {
            'root_dataset': '10% of training data (for server)',
            'client_datasets': '90% split among clients',
            'test_dataset': 'Separate test set (not used in training)',
            'split_method': 'Dirichlet distribution' if ENABLE_NON_IID else 'IID random split',
            'dirichlet_alpha': DIRICHLET_ALPHA if ENABLE_NON_IID else None
        },
        'class_balancing': {
            'method': 'Natural distribution (no oversampling)',
            'note': 'Non-IID splits may create imbalanced local datasets intentionally'
        },
        'batch_size': BATCH_SIZE,
        'data_augmentation': {
            'training': ['Random horizontal flip'] if DATASET == 'ALZHEIMER' else [],
            'validation': []
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Print documentation
    print("Dataset Preprocessing Pipeline:")
    print(f"  Dataset: {documentation['dataset']}")
    print(f"\n  Steps for {documentation['dataset']}:")
    
    if documentation['dataset'] in documentation['preprocessing_steps']:
        for step in documentation['preprocessing_steps'][documentation['dataset']]:
            print(f"    {step}")
    
    print(f"\n  Data Split:")
    for key, value in documentation['data_split'].items():
        print(f"    {key}: {value}")
    
    print(f"\n  Class Balancing:")
    for key, value in documentation['class_balancing'].items():
        print(f"    {key}: {value}")
    
    # Save JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file = output_dir / f'preprocessing_pipeline_{timestamp}.json'
    
    with open(json_file, 'w') as f:
        json.dump(documentation, f, indent=2)
    
    # Save Markdown
    md_content = f"""# Preprocessing Pipeline Documentation

**Dataset:** {documentation['dataset']}

**Generated:** {documentation['timestamp']}

## Preprocessing Steps

"""
    
    if documentation['dataset'] in documentation['preprocessing_steps']:
        for step in documentation['preprocessing_steps'][documentation['dataset']]:
            md_content += f"{step}\n"
    
    md_content += f"""
## Data Split

- **Root Dataset:** {documentation['data_split']['root_dataset']}
- **Client Datasets:** {documentation['data_split']['client_datasets']}
- **Test Dataset:** {documentation['data_split']['test_dataset']}
- **Split Method:** {documentation['data_split']['split_method']}
"""
    
    if documentation['data_split']['dirichlet_alpha']:
        md_content += f"- **Dirichlet Alpha:** {documentation['data_split']['dirichlet_alpha']}\n"
    
    md_content += f"""
## Class Balancing

- **Method:** {documentation['class_balancing']['method']}
- **Note:** {documentation['class_balancing']['note']}

## Batch Size

{documentation['batch_size']}

## Data Augmentation

**Training:** {', '.join(documentation['data_augmentation']['training']) if documentation['data_augmentation']['training'] else 'None'}

**Validation:** {', '.join(documentation['data_augmentation']['validation']) if documentation['data_augmentation']['validation'] else 'None'}
"""
    
    md_file = output_dir / f'preprocessing_pipeline_{timestamp}.md'
    with open(md_file, 'w') as f:
        f.write(md_content)
    
    print(f"\n✅ Preprocessing documentation completed!")
    print(f"📁 JSON: {json_file}")
    print(f"📄 Markdown: {md_file}")
    
    return {
        'json_file': str(json_file),
        'md_file': str(md_file),
        'documentation': documentation
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='experiments/results/preprocessing_docs')
    
    args = parser.parse_args()
    
    document_preprocessing(output_dir=args.output_dir)

