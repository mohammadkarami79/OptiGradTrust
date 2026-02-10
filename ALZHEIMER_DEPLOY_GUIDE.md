# 🧠 Alzheimer Parallel Execution - Deploy Guide

## ⚠️ CRITICAL: اجرای موازی بدون خراب کردن OASIS

تاریخ: 2026-02-05
وضعیت فعلی: OASIS در حال اجرا روی سرور (Round 2/8)

---

## 📁 فایل‌های مورد نیاز برای Alzheimer

### 1. فایل‌های اصلی (باید به‌روز باشند)
```
federated_learning/config/config_noniid_alzheimer.py   ✅ موجود
federated_learning/data/dataset.py                      ✅ تغییری نکرده
federated_learning/data/dataset_utils.py                ✅ تغییری نکرده  
federated_learning/training/server.py                   ✅ تغییری نکرده
federated_learning/training/client.py                   ✅ تغییری نکرده
federated_learning/training/aggregators.py              ✅ تغییری نکرده
federated_learning/models/vae.py                        ✅ تغییری نکرده
federated_learning/models/resnet.py                     ✅ موجود
run_all_experiments.py                                  ✅ تغییری نکرده
```

### 2. اسکریپت اجرا
```bash
run_optimized_experiments.py  # همین که الان OASIS رو اجرا میکنه
```

---

## 🚀 روش امن Deploy (گزینه 1 - توصیه میشود)

### مرحله 1: Push تغییرات Local به GitHub

```bash
# روی Windows Local
cd d:\OptiGradTrust-3
git status  # فقط .gitignore تغییر کرده
git add .gitignore federated_learning/config/config_noniid_cifar10.py
git commit -m "chore: Update config for parallel experiments"
git push origin main
```

### مرحله 2: Clone/Pull روی Server در directory جدید

```bash
# روی Server GPU
cd ~/FLBrain/

# ساخت directory جدید برای Alzheimer
mkdir -p OptiGradTrust-3-Alzheimer
cd OptiGradTrust-3-Alzheimer

# Clone کردن (اولین بار)
git clone https://github.com/YOUR_USERNAME/OptiGradTrust-3.git .

# یا Pull کردن (اگر از قبل clone شده)
git pull origin main
```

### مرحله 3: راه‌اندازی محیط موازی

```bash
# در directory جدید Alzheimer
cd ~/FLBrain/OptiGradTrust-3-Alzheimer

# فعال‌سازی environment
conda activate optigrad_py311

# بررسی dataset Alzheimer
ls -lh data/ALZHEIMER/  # مطمئن شوید data موجود است

# تست سریع configuration
python -c "from federated_learning.config.config_noniid_alzheimer import *; print(f'✅ Config loaded: {DATASET}, Epochs: {GLOBAL_EPOCHS}')"
```

### مرحله 4: اجرای Alzheimer (موازی با OASIS)

```bash
# Terminal جدید - screen session جدید
screen -S alzheimer_exp

# اجرا
nohup python run_optimized_experiments.py \
    --config federated_learning/config/config_noniid_alzheimer.py \
    --dataset ALZHEIMER \
    --epochs 8 \
    --output-dir results/alzheimer_noniid/ \
    > alzheimer_run.log 2>&1 &

# خروج از screen
Ctrl+A, D

# چک کردن log
tail -f alzheimer_run.log
```

---

## 🔍 بررسی ایمنی (قبل از اجرا)

### چک کنید:

```bash
# 1. OASIS هنوز در حال اجرا است؟
screen -ls  # باید oasis_exp رو ببینید
ps aux | grep python | grep oasis

# 2. Directory‌ها جدا هستند؟
pwd  # باید OptiGradTrust-3-Alzheimer باشد
cd ../OptiGradTrust-3  # directory قبلی OASIS
pwd  # باید OptiGradTrust-3 باشد (بدون -Alzheimer)

# 3. Output directory‌ها متفاوت هستند؟
# OASIS: results/oasis_experiments/
# Alzheimer: results/alzheimer_noniid/
```

---

## 🛡️ استراتژی امن (بدون risk)

### ✅ چرا این روش امن است:

1. **Directory‌های جدا**:
   - OASIS: `~/FLBrain/OptiGradTrust-3/`
   - Alzheimer: `~/FLBrain/OptiGradTrust-3-Alzheimer/`

2. **Output‌های جدا**:
   - OASIS results: `results/oasis_experiments/`
   - Alzheimer results: `results/alzheimer_noniid/`

3. **Screen sessions جدا**:
   - OASIS: screen session `oasis_exp`
   - Alzheimer: screen session `alzheimer_exp`

4. **Log files جدا**:
   - OASIS: `oasis_final.log`
   - Alzheimer: `alzheimer_run.log`

---

## 🚨 اگر مشکلی پیش آمد

### اگر OASIS متوقف شد:
```bash
cd ~/FLBrain/OptiGradTrust-3
screen -r oasis_exp  # Attach کردن به session
# اگر متوقف شده، دوباره resume کنید
```

### اگر Alzheimer مشکل داشت:
```bash
cd ~/FLBrain/OptiGradTrust-3-Alzheimer
screen -r alzheimer_exp
# بررسی log
tail -100 alzheimer_run.log
```

### Kill کردن Alzheimer (بدون آسیب به OASIS):
```bash
# فقط Alzheimer process را kill میکند
pkill -f "python.*alzheimer"
# یا
screen -X -S alzheimer_exp quit
```

---

## ⏱️ زمان تخمینی

- **OASIS باقیمانده**: ~10 دقیقه (Round 2/8 در حال اجرا)
- **Alzheimer اجرا**: ~15-20 دقیقه (8 rounds)
- **همزمان**: هر دو در ~25 دقیقه تمام میشوند

---

## ✅ Checklist نهایی قبل از اجرا

- [ ] Git push انجام شد (local → GitHub)
- [ ] Directory جدید ساخته شد: `OptiGradTrust-3-Alzheimer`
- [ ] Git pull/clone انجام شد
- [ ] Config file Alzheimer test شد
- [ ] Dataset Alzheimer موجود است
- [ ] Output directory متفاوت تنظیم شد
- [ ] Screen session جدید ساخته شد
- [ ] OASIS هنوز در حال اجرا است (چک شد)

---

## 📞 مانیتورینگ هر دو اجرا

```bash
# Terminal 1: OASIS monitoring
watch -n 10 'tail -20 ~/FLBrain/OptiGradTrust-3/oasis_final.log'

# Terminal 2: Alzheimer monitoring  
watch -n 10 'tail -20 ~/FLBrain/OptiGradTrust-3-Alzheimer/alzheimer_run.log'

# یا در یک terminal:
watch -n 10 'echo "=== OASIS ===" && tail -10 ~/FLBrain/OptiGradTrust-3/oasis_final.log && echo "" && echo "=== ALZHEIMER ===" && tail -10 ~/FLBrain/OptiGradTrust-3-Alzheimer/alzheimer_run.log'
```

---

## ✨ نتیجه‌گیری

این روش **کاملاً امن** است چون:
- هیچ file ای از OASIS لمس نمیشه
- Directory‌ها کاملاً جدا هستند
- هر کدوم output خودش رو داره
- میتونید هر لحظه یکی رو stop کنید بدون آسیب به دیگری

**موفق باشید! 🚀**
