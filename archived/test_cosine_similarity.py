import torch
import torch.nn.functional as F
import numpy as np

def test_cosine_similarity_with_root():
    try:
        # Test case 1: Identical vectors should have similarity 1.0
        grad1 = torch.tensor([1.0, 2.0, 3.0])
        root1 = torch.tensor([1.0, 2.0, 3.0])
        similarity1 = F.cosine_similarity(grad1, root1, dim=0).item()
        print(f"Test 1 - Identical vectors similarity: {similarity1}")
        assert np.isclose(similarity1, 1.0), f"Expected 1.0, got {similarity1}"

        # Test case 2: Orthogonal vectors should have similarity 0.0
        grad2 = torch.tensor([1.0, 0.0, 0.0])
        root2 = torch.tensor([0.0, 1.0, 0.0])
        similarity2 = F.cosine_similarity(grad2, root2, dim=0).item()
        print(f"Test 2 - Orthogonal vectors similarity: {similarity2}")
        assert np.isclose(similarity2, 0.0), f"Expected 0.0, got {similarity2}"

        # Test case 3: Opposite vectors should have similarity -1.0
        grad3 = torch.tensor([1.0, 2.0, 3.0])
        root3 = torch.tensor([-1.0, -2.0, -3.0])
        similarity3 = F.cosine_similarity(grad3, root3, dim=0).item()
        print(f"Test 3 - Opposite vectors similarity: {similarity3}")
        assert np.isclose(similarity3, -1.0), f"Expected -1.0, got {similarity3}"

        # Test case 4: Real-world like gradients
        grad4 = torch.tensor([0.15, -0.23, 0.47, 0.01, -0.32])
        root4 = torch.tensor([0.12, -0.20, 0.45, 0.02, -0.30])
        similarity4 = F.cosine_similarity(grad4, root4, dim=0).item()
        print(f"Test 4 - Real-world like gradients similarity: {similarity4}")
        # Should be close to 1.0 as vectors are similar
        assert similarity4 > 0.99, f"Expected > 0.99, got {similarity4}"

        # Test case 5: Different magnitudes but same direction
        grad5 = torch.tensor([2.0, 4.0, 6.0])
        root5 = torch.tensor([1.0, 2.0, 3.0])
        similarity5 = F.cosine_similarity(grad5, root5, dim=0).item()
        print(f"Test 5 - Same direction, different magnitude similarity: {similarity5}")
        assert np.isclose(similarity5, 1.0), f"Expected 1.0, got {similarity5}"

        print("All cosine similarity tests passed!")
    except Exception as e:
        print(f"Error occurred during testing: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting cosine similarity tests...")
    test_cosine_similarity_with_root() 