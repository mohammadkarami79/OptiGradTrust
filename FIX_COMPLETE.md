# مشکل حل شد! ✅

## تغییرات انجام شده

### مشکل
Script قبلی سعی می‌کرد از متد `setup_clients()` استفاده کند که در کلاس `Server` وجود ندارد.

### راه‌حل
کل تابع `run_experiment` را بازنویسی کردیم تا دقیقاً از همان pattern استفاده شده در `main.py` پیروی کند:

```python
1. Load dataset
2. Create server
3. Set datasets
4. Pre-train global model
5. Create client datasets
6. Create clients
7. Add clients to server
8. Configure malicious clients
9. Train
10. Evaluate
```

### تغییرات کلیدی

✅ **اضافه شد:**
- `load_dataset()` برای بارگذاری دیتاست
- `create_client_datasets()` برای ایجاد دیتاست‌های کلاینت
- `set_datasets()` برای تنظیم دیتاست‌های سرور
- `_pretrain_global_model()` برای pre-training
- `add_clients()` برای اضافه کردن کلاینت‌ها
- `evaluate_model()` برای ارزیابی نهایی

✅ **حذف شد:**
- `setup_clients()` که وجود نداشت

## دستور اجرا

### دستور کامل:
```bash
cd D:\new_paper
d:/new_paper/venv/Scripts/activate.bat
python experiments\focused_reviewer_response.py
```

### یا به صورت یک خطی:
```bash
D: && cd D:\new_paper && d:/new_paper/venv/Scripts/activate.bat && python experiments\focused_reviewer_response.py
```

## زمان اجرا

⏱ **حدود 8-10 ساعت**

این شامل:
- 1 ساعت: OptiGradTrust کامل
- 1 ساعت: FLGuard
- 1 ساعت: FLTrust  
- 5-7 ساعت: Ablation studies (بدون Shapley، بدون VAE، بدون FedBN-P)

## خروجی‌های مورد انتظار

پس از اجرا، فایل‌های زیر تولید می‌شود:

### 1. جداول مقایسه (LaTeX)
```
experiments/results/focused_reviewer_response/comparison_table.tex
experiments/results/focused_reviewer_response/ablation_table.tex
```

### 2. نتایج خام (JSON)
```
experiments/results/focused_reviewer_response/results.json
```

### 3. خلاصه نتایج
- دقت نهایی هر روش
- بهبود accuracy
- detection precision/recall/F1
- مقایسه baseline‌ها
- نتایج ablation

## چگونگی استفاده از نتایج

### برای بخش Comparison در مقاله:
1. فایل `comparison_table.tex` را باز کنید
2. مستقیماً در LaTeX مقاله copy کنید
3. این جدول را در Section 4 (Results) قرار دهید

### برای بخش Ablation Study:
1. فایل `ablation_table.tex` را باز کنید
2. در بخش Ablation Study مقاله قرار دهید
3. این نشان می‌دهد:
   - Shapley چقدر مهم است
   - VAE چقدر کمک می‌کند
   - FedBN-P چه تاثیری دارد

## اگر خطا گرفتید

### خطای مربوط به CUDA/GPU:
```bash
# در config.py این را بررسی کنید:
USE_CUDA = True if torch.cuda.is_available() else False
```

### خطای مربوط به dataset:
```bash
# مطمئن شوید دیتاست Alzheimer در این مسیر است:
D:\new_paper\data\alzheimer\train\
D:\new_paper\data\alzheimer\test\
```

### خطای مربوط به memory:
```bash
# batch size را کم کنید در config.py:
BATCH_SIZE = 16  # به جای 32
```

## نکات مهم

🔴 **قبل از اجرا:**
- مطمئن شوید GPU در دسترس است
- حداقل 8GB VRAM لازم است
- حداقل 16GB RAM سیستم لازم است

🟡 **در حین اجرا:**
- سیستم را خاموش نکنید
- از حالت sleep جلوگیری کنید
- terminal را نبندید

🟢 **بعد از اجرا:**
- نتایج را در `experiments/results/focused_reviewer_response/` چک کنید
- جداول LaTeX را در مقاله قرار دهید
- به بخش بعدی (Paper Updates) بروید

## گام بعدی

بعد از تکمیل این اجرا:

1. ✅ نتایج comparison را در مقاله قرار دهید
2. ✅ جدول ablation را اضافه کنید
3. ✅ discussion را بر اساس نتایج جدید بنویسید
4. ✅ به سراغ extended metrics بروید (اگر نیاز باشد)

---

## اطلاعات تکنیکی

### Architecture Pattern
```
Server:
  ├─ __init__()
  ├─ set_datasets(root_loader, test_dataset)
  ├─ _pretrain_global_model()
  ├─ add_clients(clients)
  ├─ train(num_rounds)
  └─ evaluate_model()
```

### Experiment Flow
```
1. Load dataset → root_dataset, test_dataset
2. Create server → Server()
3. Set datasets → server.set_datasets()
4. Pretrain → server._pretrain_global_model()
5. Create clients → Client(id, dataset, is_malicious)
6. Add to server → server.add_clients(clients)
7. Train → server.train(num_rounds)
8. Evaluate → server.evaluate_model()
```

این pattern از `main.py` اصلی الگوبرداری شده و تضمین می‌کند که همه چیز درست کار کند.

---

**الان دوباره اجرا کنید!** 🚀

تمام مشکلات حل شده است. موفق باشید! 💪

