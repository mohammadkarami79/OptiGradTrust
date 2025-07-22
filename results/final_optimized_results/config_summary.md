# تنظیمات بهینه نهایی برای هر Dataset
============================================

## 🎯 **MNIST + CNN - Config بهینه:**
```python
DATASET = 'MNIST'
MODEL = 'CNN'
BATCH_SIZE = 32
LEARNING_RATE = 0.01
LOCAL_EPOCHS_ROOT = 30
LOCAL_EPOCHS_CLIENT = 8
GLOBAL_EPOCHS = 20

# Detection Parameters
MALICIOUS_PENALTY_FACTOR = 0.4
ZERO_ATTACK_THRESHOLD = 0.005
HIGH_GRADIENT_STD_MULTIPLIER = 2.5
CONFIDENCE_THRESHOLD = 0.7
DUAL_ATTENTION_THRESHOLD = 0.65
```

## 💪 **CIFAR-10 + ResNet18 - Config بهینه:**
```python
DATASET = 'CIFAR10'
MODEL = 'RESNET18'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LOCAL_EPOCHS_ROOT = 35
LOCAL_EPOCHS_CLIENT = 6
GLOBAL_EPOCHS = 30

# Detection Parameters
MALICIOUS_PENALTY_FACTOR = 0.35
ZERO_ATTACK_THRESHOLD = 0.002
HIGH_GRADIENT_STD_MULTIPLIER = 3.0
CONFIDENCE_THRESHOLD = 0.75
DUAL_ATTENTION_THRESHOLD = 0.7
```

## 🏆 **ALZHEIMER + ResNet18 - Config بهینه:**
```python
DATASET = 'ALZHEIMER'
MODEL = 'RESNET18'
BATCH_SIZE = 24
LEARNING_RATE = 0.003
LOCAL_EPOCHS_ROOT = 40
LOCAL_EPOCHS_CLIENT = 7
GLOBAL_EPOCHS = 30

# Detection Parameters  
MALICIOUS_PENALTY_FACTOR = 0.25
ZERO_ATTACK_THRESHOLD = 0.001
HIGH_GRADIENT_STD_MULTIPLIER = 4.0
CONFIDENCE_THRESHOLD = 0.85
DUAL_ATTENTION_THRESHOLD = 0.8
```

## 🔍 **توضیح تنظیمات:**

### **پارامترهای مشترک:**
- **SHAPLEY_WEIGHT**: 0.6-0.8 (بسته به dataset)
- **VAE_DEVICE**: 'cpu' (بهینه‌سازی حافظه)
- **ENABLE_VAE**: True
- **ENABLE_SHAPLEY**: True
- **ENABLE_DUAL_ATTENTION**: True

### **منطق بهینه‌سازی:**
1. **MNIST**: تمرکز بر کاهش false positives
2. **CIFAR-10**: متعادل کردن accuracy و detection
3. **ALZHEIMER**: ماکزیمم کردن precision برای medical

---
*این تنظیمات براساس آزمایش و تحلیل علمی بهینه شده‌اند* 