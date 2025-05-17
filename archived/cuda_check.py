import os
import sys
import subprocess
import torch

def run_command(command):
    """اجرای یک دستور در خط فرمان و بازگرداندن خروجی آن"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

print("="*50)
print("CUDA و PyTorch بررسی تنظیمات")
print("="*50)

# بررسی نسخه Python
print(f"Python version: {sys.version}")

# بررسی نسخه PyTorch
print(f"PyTorch version: {torch.__version__}")

# بررسی پشتیبانی CUDA در PyTorch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version (if available): {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Not available'}")

# بررسی GPU های موجود از طریق NVIDIA-SMI
print("\nGPU Information from nvidia-smi:")
nvidia_smi = run_command("nvidia-smi")
print(nvidia_smi)

# بررسی متغیرهای محیطی مرتبط با CUDA
print("\nCUDA Environment Variables:")
cuda_path = os.environ.get('CUDA_PATH', 'Not set')
path = os.environ.get('PATH', 'Not set')
print(f"CUDA_PATH: {cuda_path}")
print(f"PATH contains CUDA: {'cuda' in path.lower()}")

# راه حل پیشنهادی
print("\n"+"="*50)
print("راه حل های پیشنهادی:")
print("="*50)

if not torch.cuda.is_available():
    if "NVIDIA" in nvidia_smi:
        print("1. GPU شناسایی شده است ولی PyTorch نمی‌تواند از آن استفاده کند.")
        print("2. نصب مجدد PyTorch با پشتیبانی CUDA:")
        print("   pip uninstall torch")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("   (جایگزین cu121 با نسخه CUDA خود - مثلاً cu118 برای CUDA 11.8)")
        print("3. اطمینان حاصل کنید که نسخه CUDA در PyTorch با نسخه درایور NVIDIA شما سازگار است.")
    else:
        print("1. هیچ GPU NVIDIA شناسایی نشد. لطفاً موارد زیر را بررسی کنید:")
        print("   - درایورهای NVIDIA به درستی نصب شده‌اند")
        print("   - GPU NVIDIA روی این سیستم موجود است")
        print("2. اگر نمی‌خواهید از GPU استفاده کنید، تنظیم FORCE_GPU را در config.py به False تغییر دهید.")
else:
    print("CUDA به درستی کار می‌کند!")
    
print("\nبرای تست سریع عملکرد GPU و CUDA:")
print("python -c \"import torch; x = torch.rand(5, 3).cuda(); print(x)\"")

# بررسی آیا با تنظیم دستی device می‌توان روی GPU کار کرد
print("\nتلاش برای استفاده از GPU به صورت دستی:")
try:
    test_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Tensor created on device: {test_tensor.device}")
    if 'cuda' in str(test_tensor.device):
        print("موفقیت! می‌توان از GPU استفاده کرد.")
    else:
        print("تنسور روی CPU ایجاد شد.")
except Exception as e:
    print(f"خطا در ایجاد تنسور: {str(e)}") 