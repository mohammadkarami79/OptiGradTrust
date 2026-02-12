"""
OASIS-1 Cross-Sectional MRI Dataset Loader

This module provides data loading functionality for the OASIS-1 (Open Access Series of Imaging Studies)
cross-sectional MRI dataset for Alzheimer's disease classification.

Dataset Information:
- 416 subjects aged 18-96
- 100 subjects over 60 diagnosed with very mild to moderate AD
- 3-4 T1-weighted MRI scans per subject

Classification based on CDR (Clinical Dementia Rating):
- CDR 0: Nondemented (Class 0)
- CDR 0.5: Very Mild Dementia (Class 1)
- CDR 1: Mild Dementia (Class 2)
- CDR 2: Moderate Dementia (Class 3)

Reference:
Marcus, DS, Wang, TH, Parker, J, Csernansky, JG, Morris, JC, Buckner, RL.
"Open Access Series of Imaging Studies (OASIS): Cross-Sectional MRI Data in Young,
Middle Aged, Nondemented, and Demented Older Adults"
Journal of Cognitive Neuroscience, 19, 1498-1507. doi: 10.1162/jocn.2007.19.9.1498
"""

import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from typing import Optional, Tuple, List, Dict
import warnings
import urllib.request
import re


# OASIS-1 dataset configuration
OASIS_IMG_SIZE = 224  # Standard size for ResNet
OASIS_CLASSES = 4  # Nondemented, Very Mild, Mild, Moderate

# CDR to class mapping
CDR_TO_CLASS = {
    0.0: 0,   # Nondemented
    0.5: 1,   # Very Mild Dementia
    1.0: 2,   # Mild Dementia
    2.0: 3,   # Moderate Dementia
}

CLASS_NAMES = [
    'Nondemented',
    'VeryMildDementia',
    'MildDementia', 
    'ModerateDementia'
]


class OASISDataset(Dataset):
    """
    PyTorch Dataset for OASIS-1 Cross-Sectional MRI data.
    
    Loads processed MRI images (GIF format) from the OASIS-1 directory structure
    and assigns labels based on Clinical Dementia Rating (CDR) scores.
    
    Args:
        root_dir: Path to OASIS disc directory (e.g., 'oasis_cross-sectional_disc1/disc1')
        demographics_csv: Path to OASIS demographics CSV file (optional)
        transform: Optional image transforms
        image_type: Type of processed image to use ('tra', 'sag', 'cor' for transverse, sagittal, coronal)
        use_masked: Whether to use masked (brain-extracted) images
        split: 'train', 'test', or 'all'
        train_ratio: Ratio of data for training (default 0.8)
        random_seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        root_dir: str,
        demographics_csv: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        image_type: str = 'tra',
        use_masked: bool = True,
        split: str = 'all',
        train_ratio: float = 0.8,
        random_seed: int = 42
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.image_type = image_type
        self.use_masked = use_masked
        self.split = split
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        
        # Load demographics data
        self.demographics = self._load_demographics(demographics_csv)
        
        # Find all subject directories
        self.subjects = self._find_subjects()
        
        # Build image paths and labels
        self.samples = self._build_samples()
        
        # Apply train/test split
        if split != 'all':
            self._apply_split()
        
        print(f"OASIS Dataset initialized: {len(self.samples)} samples")
        self._print_class_distribution()
    
    def _load_demographics(self, csv_path: Optional[str]) -> pd.DataFrame:
        """Load demographics from CSV or Excel file, or create synthetic labels."""
        
        def _read_file(path: str) -> Optional[pd.DataFrame]:
            """Read a CSV or Excel file based on extension."""
            if path.endswith('.xlsx') or path.endswith('.xls'):
                print(f"Loading demographics from Excel file: {path}")
                try:
                    df = pd.read_excel(path)
                    return df
                except Exception as e:
                    print(f"Warning: Failed to read Excel file {path}: {e}")
                    return None
            else:
                print(f"Loading demographics from CSV file: {path}")
                try:
                    df = pd.read_csv(path)
                    return df
                except Exception as e:
                    print(f"Warning: Failed to read CSV file {path}: {e}")
                    return None
        
        # First try the explicitly provided path
        if csv_path and os.path.exists(csv_path):
            df = _read_file(csv_path)
            if df is not None:
                return df
        
        # Try to find demographics file in common locations
        # Support both CSV and Excel formats
        base_names = ['oasis_cross-sectional']
        extensions = ['.xlsx', '.xls', '.csv']
        search_dirs = [
            os.path.dirname(self.root_dir),
            os.path.dirname(os.path.dirname(self.root_dir)),
            'data/oasis',
            '.',  # Current directory
        ]
        
        possible_paths = []
        for search_dir in search_dirs:
            for base_name in base_names:
                for ext in extensions:
                    possible_paths.append(os.path.join(search_dir, base_name + ext))
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found demographics at: {path}")
                df = _read_file(path)
                if df is not None:
                    return df
        
        print("WARNING: Demographics file not found (tried CSV and Excel formats).")
        print("Using synthetic labels based on OASIS-1 distribution.")
        print("For accurate results, download demographics from: https://www.oasis-brains.org/")
        print("Place the file as 'oasis_cross-sectional.xlsx' or 'oasis_cross-sectional.csv' in the project root.")
        return None
    
    def _find_subjects(self) -> List[str]:
        """Find all subject directories in the OASIS disc."""
        subjects = []
        
        # Handle different possible directory structures
        if os.path.exists(self.root_dir):
            # Look for OAS1_XXXX_MR1 pattern directories
            for item in os.listdir(self.root_dir):
                if item.startswith('OAS1_') and os.path.isdir(os.path.join(self.root_dir, item)):
                    subjects.append(item)
        
        # Also check for disc subdirectory
        disc_dir = os.path.join(self.root_dir, 'disc1')
        if os.path.exists(disc_dir):
            for item in os.listdir(disc_dir):
                if item.startswith('OAS1_') and os.path.isdir(os.path.join(disc_dir, item)):
                    subjects.append(item)
            self.root_dir = disc_dir  # Update root dir
        
        subjects = sorted(list(set(subjects)))
        print(f"Found {len(subjects)} subjects")
        return subjects
    
    def _get_cdr_for_subject(self, subject_id: str) -> float:
        """Get CDR score for a subject from demographics or synthetic assignment."""
        
        # Extract numeric ID from subject folder name (e.g., OAS1_0001_MR1 -> 0001)
        match = re.search(r'OAS1_(\d+)_MR', subject_id)
        if not match:
            return 0.0  # Default to nondemented
        
        numeric_id = int(match.group(1))
        
        # If we have demographics data, use it
        if self.demographics is not None:
            try:
                # Try different column name formats
                id_col = None
                for col in ['ID', 'Subject', 'Subject ID', 'OAS_ID']:
                    if col in self.demographics.columns:
                        id_col = col
                        break
                
                if id_col:
                    # Try to match subject
                    row = self.demographics[
                        self.demographics[id_col].str.contains(f'OAS1_{numeric_id:04d}', na=False)
                    ]
                    
                    if not row.empty and 'CDR' in self.demographics.columns:
                        cdr = row['CDR'].values[0]
                        if pd.notna(cdr):
                            return float(cdr)
            except Exception as e:
                pass
        
        # Synthetic CDR assignment based on OASIS-1 statistics:
        # - ~316 subjects are nondemented (76%)
        # - ~70 subjects have very mild dementia (17%)
        # - ~28 subjects have mild dementia (7%)
        # - ~2 subjects have moderate dementia (0.5%)
        
        # Use deterministic assignment based on subject ID for reproducibility
        np.random.seed(numeric_id + self.random_seed)
        rand_val = np.random.random()
        
        if rand_val < 0.76:
            return 0.0   # Nondemented
        elif rand_val < 0.93:
            return 0.5   # Very Mild
        elif rand_val < 0.995:
            return 1.0   # Mild
        else:
            return 2.0   # Moderate
    
    def _find_image_for_subject(self, subject_id: str) -> Optional[str]:
        """Find the appropriate processed MRI image for a subject."""
        
        subject_dir = os.path.join(self.root_dir, subject_id)
        
        # Preferred path: PROCESSED/MPRAGE/T88_111/
        processed_dir = os.path.join(subject_dir, 'PROCESSED', 'MPRAGE', 'T88_111')
        
        if os.path.exists(processed_dir):
            # Build search pattern
            if self.use_masked:
                pattern = f"*masked*{self.image_type}*.gif"
            else:
                pattern = f"*gfc_{self.image_type}*.gif"
            
            # Find matching files
            matches = glob.glob(os.path.join(processed_dir, pattern))
            
            if matches:
                return matches[0]
            
            # Fallback: any GIF in the processed directory
            all_gifs = glob.glob(os.path.join(processed_dir, "*.gif"))
            if all_gifs:
                return all_gifs[0]
        
        # Alternative: FSL_SEG directory
        fsl_dir = os.path.join(subject_dir, 'FSL_SEG')
        if os.path.exists(fsl_dir):
            gifs = glob.glob(os.path.join(fsl_dir, "*.gif"))
            if gifs:
                return gifs[0]
        
        # Alternative: SUBJ_111 directory  
        subj_dir = os.path.join(subject_dir, 'PROCESSED', 'MPRAGE', 'SUBJ_111')
        if os.path.exists(subj_dir):
            gifs = glob.glob(os.path.join(subj_dir, "*.gif"))
            if gifs:
                return gifs[0]
        
        # Last resort: RAW directory
        raw_dir = os.path.join(subject_dir, 'RAW')
        if os.path.exists(raw_dir):
            gifs = glob.glob(os.path.join(raw_dir, "*.gif"))
            if gifs:
                return gifs[0]
        
        return None
    
    def _build_samples(self) -> List[Tuple[str, int]]:
        """Build list of (image_path, label) tuples."""
        samples = []
        
        for subject_id in self.subjects:
            # Get image path
            img_path = self._find_image_for_subject(subject_id)
            if img_path is None:
                continue
            
            # Get CDR and convert to class
            cdr = self._get_cdr_for_subject(subject_id)
            label = CDR_TO_CLASS.get(cdr, 0)
            
            samples.append((img_path, label, subject_id))
        
        return samples
    
    def _apply_split(self):
        """Apply train/test split."""
        np.random.seed(self.random_seed)
        
        # Group by subject to avoid data leakage
        subject_samples = {}
        for img_path, label, subject_id in self.samples:
            base_subject = subject_id.rsplit('_', 1)[0]  # Remove _MR1 suffix
            if base_subject not in subject_samples:
                subject_samples[base_subject] = []
            subject_samples[base_subject].append((img_path, label, subject_id))
        
        # Split subjects
        subjects = list(subject_samples.keys())
        np.random.shuffle(subjects)
        
        split_idx = int(len(subjects) * self.train_ratio)
        
        if self.split == 'train':
            selected_subjects = subjects[:split_idx]
        else:  # test
            selected_subjects = subjects[split_idx:]
        
        # Rebuild samples
        self.samples = []
        for subj in selected_subjects:
            self.samples.extend(subject_samples[subj])
    
    def _print_class_distribution(self):
        """Print class distribution statistics."""
        class_counts = [0] * OASIS_CLASSES
        for _, label, _ in self.samples:
            class_counts[label] += 1
        
        print("\nClass Distribution:")
        for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
            pct = 100 * count / len(self.samples) if self.samples else 0
            print(f"  {name}: {count} ({pct:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label, _ = self.samples[idx]
        
        # Load image
        image = Image.open(img_path)
        
        # Convert grayscale to RGB (for ResNet compatibility)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_subject_info(self, idx: int) -> Dict:
        """Get detailed info for a sample."""
        img_path, label, subject_id = self.samples[idx]
        return {
            'subject_id': subject_id,
            'image_path': img_path,
            'label': label,
            'class_name': CLASS_NAMES[label]
        }


def get_oasis_transforms(train: bool = True, img_size: int = OASIS_IMG_SIZE):
    """Get standard transforms for OASIS dataset."""
    
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def load_oasis_dataset(
    root_dir: str,
    demographics_csv: Optional[str] = None,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset, int, int]:
    """
    Load OASIS-1 dataset with train/test split.
    
    Args:
        root_dir: Path to OASIS data directory
        demographics_csv: Path to demographics CSV (optional)
        train_ratio: Ratio for train/test split
        random_seed: Random seed for reproducibility
    
    Returns:
        train_dataset, test_dataset, num_classes, input_channels
    """
    
    print(f"\n=== Loading OASIS-1 Dataset ===")
    print(f"Root directory: {root_dir}")
    
    # Create train dataset
    train_dataset = OASISDataset(
        root_dir=root_dir,
        demographics_csv=demographics_csv,
        transform=get_oasis_transforms(train=True),
        split='train',
        train_ratio=train_ratio,
        random_seed=random_seed
    )
    
    # Create test dataset
    test_dataset = OASISDataset(
        root_dir=root_dir,
        demographics_csv=demographics_csv,
        transform=get_oasis_transforms(train=False),
        split='test',
        train_ratio=train_ratio,
        random_seed=random_seed
    )
    
    print(f"\nDataset loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: {OASIS_CLASSES}")
    print(f"  Input channels: 3 (RGB)")
    
    return train_dataset, test_dataset, OASIS_CLASSES, 3


def download_oasis_demographics():
    """
    Instructions for downloading OASIS-1 demographics data.
    """
    print("\n" + "="*60)
    print("OASIS-1 Demographics Data Download Instructions")
    print("="*60)
    print("""
For accurate CDR (Clinical Dementia Rating) labels, please download
the demographics CSV from the OASIS website:

1. Visit: https://www.oasis-brains.org/
2. Register for an account (free)
3. Download "OASIS-1: Cross-Sectional: Demographic and Clinical Data"
4. Save as 'oasis_cross-sectional.csv' in the data/oasis/ directory

The demographics file contains:
- Subject ID
- M/F (gender)
- Hand (handedness)
- Age
- Educ (years of education)
- SES (socioeconomic status)
- MMSE (Mini-Mental State Examination)
- CDR (Clinical Dementia Rating) <- Used for classification
- eTIV (estimated total intracranial volume)
- nWBV (normalized whole brain volume)
- ASF (Atlas Scaling Factor)

Without this file, synthetic labels will be generated based on
the published OASIS-1 distribution statistics.
""")
    print("="*60 + "\n")


# Test function
def test_oasis_dataset(root_dir: str):
    """Test OASIS dataset loading."""
    print("\n" + "="*60)
    print("Testing OASIS Dataset")
    print("="*60)
    
    try:
        train_ds, test_ds, num_classes, in_channels = load_oasis_dataset(root_dir)
        
        # Test loading a sample
        if len(train_ds) > 0:
            img, label = train_ds[0]
            print(f"\nSample test:")
            print(f"  Image shape: {img.shape}")
            print(f"  Label: {label} ({CLASS_NAMES[label]})")
            print(f"  Subject info: {train_ds.get_subject_info(0)}")
        
        # Create a dataloader test
        loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
        batch_imgs, batch_labels = next(iter(loader))
        print(f"\nBatch test:")
        print(f"  Batch images shape: {batch_imgs.shape}")
        print(f"  Batch labels: {batch_labels.tolist()}")
        
        print("\n[SUCCESS] OASIS dataset test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] OASIS dataset test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Default test path
    import sys
    
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "oasis_cross-sectional_disc1/disc1"
    
    download_oasis_demographics()
    test_oasis_dataset(root_dir)
