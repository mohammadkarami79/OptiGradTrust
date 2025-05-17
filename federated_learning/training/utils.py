def test(model, test_loader):
    """Test model performance with detailed metrics."""
    device = next(model.parameters()).device
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * 10  # Assuming 10 classes
    class_total = [0] * 10
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Calculate per-class accuracy
            for i in range(target.size(0)):
                label = target[i]
                pred_label = pred[i]
                if pred_label == label:
                    class_correct[label] += 1
                class_total[label] += 1
    
    # Calculate overall metrics
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / total
    error_rate = 100. - accuracy
    
    # Calculate per-class metrics
    class_accuracies = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                       for i in range(len(class_correct))]
    
    # Print detailed results
    print("\nTest Results:")
    print(f"Average loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Error rate: {error_rate:.2f}%")
    print("\nPer-class accuracy:")
    for i in range(len(class_accuracies)):
        print(f"Class {i}: {class_accuracies[i]:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return accuracy / 100.0, error_rate / 100.0  # Return as decimals 