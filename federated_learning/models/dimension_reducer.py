import torch
import numpy as np
from sklearn.decomposition import PCA
import time

class GradientDimensionReducer:
    """
    A class that reduces the dimensionality of gradients using PCA.
    
    This helps to reduce memory consumption while preserving important features in gradients.
    The reduction ratio controls how much information is retained (1.0 = no reduction, 
    smaller values = more aggressive reduction).
    """
    
    def __init__(self, reduction_ratio=1.0):
        """
        Initialize the gradient dimension reducer.
        
        Args:
            reduction_ratio (float): The ratio of dimensions to keep (0.0-1.0)
                                   1.0 means no reduction, 0.5 means keep 50% of dimensions
        """
        self.reduction_ratio = reduction_ratio
        self.pca = None
        self.n_components = None
        self.original_shapes = {}
        self.is_fitted = False
        
    def fit(self, gradients):
        """
        Fit PCA to the provided gradients.
        
        Args:
            gradients: A list of tensors or a single tensor representing gradients
            
        Returns:
            self: The fitted reducer
        """
        if self.reduction_ratio >= 1.0:
            print("GradientDimensionReducer: No reduction applied (ratio >= 1.0)")
            self.is_fitted = True
            return self
            
        # Handle list of tensors or single tensor
        if isinstance(gradients, list):
            # Store original shapes for reconstruction
            self.original_shapes = {i: grad.shape for i, grad in enumerate(gradients)}
            
            # Convert gradients to numpy and flatten
            flat_grads = []
            for grad in gradients:
                if isinstance(grad, torch.Tensor):
                    flat_grads.append(grad.detach().cpu().numpy().flatten())
                else:
                    flat_grads.append(grad.flatten())
                    
            # Stack flattened gradients
            all_grads = np.vstack(flat_grads)
        else:
            # Single tensor/array case
            self.original_shapes = {0: gradients.shape}
            if isinstance(gradients, torch.Tensor):
                all_grads = gradients.detach().cpu().numpy().reshape(1, -1)
            else:
                all_grads = gradients.reshape(1, -1)
        
        # Calculate how many components to keep
        max_components = min(all_grads.shape)
        self.n_components = max(1, int(max_components * self.reduction_ratio))
        
        print(f"GradientDimensionReducer: Fitting PCA with {self.n_components} components")
        print(f"Original dimensions: {all_grads.shape[1]}, Reduced dimensions: {self.n_components}")
        print(f"Reduction ratio: {self.reduction_ratio:.2f}")
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(all_grads)
        
        # Print explained variance information
        explained_variance_ratio_sum = self.pca.explained_variance_ratio_.sum()
        print(f"Explained variance ratio sum: {explained_variance_ratio_sum:.4f}")
        
        self.is_fitted = True
        return self
        
    def transform(self, gradients):
        """
        Transform gradients to reduced dimensions.
        
        Args:
            gradients: A list of tensors or a single tensor representing gradients
            
        Returns:
            reduced_gradients: Gradients with reduced dimensions in the same format as input
        """
        if self.reduction_ratio >= 1.0 or not self.is_fitted:
            return gradients
            
        # Check if input is a list of tensors or a single tensor
        is_list = isinstance(gradients, list)
        is_tensor = isinstance(gradients if not is_list else gradients[0], torch.Tensor)
        
        # Process the gradients
        if is_list:
            # Convert each gradient tensor to numpy and flatten
            flat_grads = []
            for grad in gradients:
                if isinstance(grad, torch.Tensor):
                    flat_grads.append(grad.detach().cpu().numpy().flatten())
                else:
                    flat_grads.append(grad.flatten())
                    
            # Transform using PCA
            reduced_flat = self.pca.transform(np.vstack(flat_grads))
            
            # Convert back to original format
            result = []
            for i, grad in enumerate(gradients):
                reduced = reduced_flat[i].reshape(1, -1)
                if is_tensor:
                    result.append(torch.tensor(reduced, device=grad.device, dtype=grad.dtype))
                else:
                    result.append(reduced)
            return result
        else:
            # Single tensor/array case
            if is_tensor:
                flat_grad = gradients.detach().cpu().numpy().reshape(1, -1)
            else:
                flat_grad = gradients.reshape(1, -1)
                
            # Transform using PCA
            reduced = self.pca.transform(flat_grad)
            
            # Convert back to original format
            if is_tensor:
                return torch.tensor(reduced, device=gradients.device, dtype=gradients.dtype)
            else:
                return reduced
    
    def inverse_transform(self, reduced_gradients):
        """
        Reconstruct original gradients from reduced gradients.
        
        Args:
            reduced_gradients: A list of tensors or a single tensor with reduced dimensions
            
        Returns:
            reconstructed_gradients: Reconstructed gradients in original dimensions
        """
        if self.reduction_ratio >= 1.0 or not self.is_fitted:
            return reduced_gradients
            
        # Check if input is a list or a single tensor
        is_list = isinstance(reduced_gradients, list)
        is_tensor = isinstance(reduced_gradients if not is_list else reduced_gradients[0], torch.Tensor)
        
        # Process the reduced gradients
        if is_list:
            # Convert each reduced gradient to numpy
            reduced_flat = []
            for grad in reduced_gradients:
                if isinstance(grad, torch.Tensor):
                    reduced_flat.append(grad.detach().cpu().numpy().reshape(1, -1))
                else:
                    reduced_flat.append(grad.reshape(1, -1))
                    
            # Inverse transform using PCA
            reconstructed_flat = self.pca.inverse_transform(np.vstack(reduced_flat))
            
            # Reshape to original shapes and convert back to original format
            result = []
            for i, grad in enumerate(reduced_gradients):
                original_shape = self.original_shapes.get(i, None)
                if original_shape is None:
                    # If shape not found, keep as flattened
                    reconstructed = reconstructed_flat[i].reshape(1, -1)
                else:
                    reconstructed = reconstructed_flat[i].reshape(original_shape)
                    
                if is_tensor:
                    result.append(torch.tensor(reconstructed, device=grad.device, dtype=grad.dtype))
                else:
                    result.append(reconstructed)
            return result
        else:
            # Single tensor/array case
            if is_tensor:
                reduced_flat = reduced_gradients.detach().cpu().numpy().reshape(1, -1)
            else:
                reduced_flat = reduced_gradients.reshape(1, -1)
                
            # Inverse transform using PCA
            reconstructed = self.pca.inverse_transform(reduced_flat)
            
            # Reshape to original shape
            original_shape = self.original_shapes.get(0, None)
            if original_shape is not None:
                reconstructed = reconstructed.reshape(original_shape)
                
            # Convert back to original format
            if is_tensor:
                return torch.tensor(reconstructed, device=reduced_gradients.device, dtype=reduced_gradients.dtype)
            else:
                return reconstructed 