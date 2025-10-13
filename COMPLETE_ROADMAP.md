# 🗺️ نقشه راه کامل تا تکمیل پروژه

## 🎯 هدف نهایی
تولید نتایج معتبر و قابل اعتماد برای پاسخ به reviewer ها و ویرایش مقاله

---

## ⚠️ درس گرفته شده از اجرای قبلی

**مشکل:** اجرای 10 ساعته روی MNIST (بی‌ارزش) به جای Alzheimer!

**راه‌حل:** تست 3 مرحله‌ای قبل از اجرای نهایی

---

## 📋 مرحله 1: Quick Test (5-10 دقیقه) ⏱️

### هدف:
تأیید اولیه که:
- ✅ Alzheimer dataset لود می‌شود
- ✅ Ablation واقعاً کار می‌کند
- ✅ نتایج متفاوت هستند

### دستور:
```bash
cd D:\new_paper
d:/new_paper/venv/Scripts/activate.bat
python experiments\test_quick_verify.py
```

### چه چیزی test می‌شود:
- 2 experiment (Full + Without Shapley)
- 3 rounds فقط
- Attack: partial_scaling

### نتیجه مورد انتظار:
```
✅ CHECK 1: Dataset is Alzheimer
✅ CHECK 2: Accuracy in reasonable range (not MNIST)
✅ CHECK 3: Results are different
```

### اگر موفق شد:
➡️ برو به مرحله 2

### اگر failed شد:
❌ مشکل را حل کن، دوباره اجرا کن

---

## 📋 مرحله 2: Medium Test (10-15 دقیقه) ⏱️

### هدف:
تأیید جامع‌تر:
- ✅ همه ablation ها کار می‌کنند
- ✅ FedAvg baseline تفاوت دارد
- ✅ Multiple attacks کار می‌کنند

### دستور:
```bash
python experiments\test_medium_verify.py
```

### چه چیزی test می‌شود:
- 5 experiments:
  1. Full OptiGradTrust
  2. Without Shapley
  3. Without VAE
  4. FedAvg Baseline
  5. Label Flipping Attack
- 10 rounds
- 2 attack types

### نتیجه مورد انتظار:
```
✅ CHECK 1: Shapley makes a difference
✅ CHECK 2: VAE makes a difference
✅ CHECK 3: OptiGradTrust beats FedAvg
✅ CHECK 4: Accuracy in Alzheimer range
✅ CHECK 5: Different attacks work
```

### اگر موفق شد (4/5 یا بیشتر):
➡️ برو به مرحله 3 (Full Run)

### اگر failed شد:
❌ تحلیل کن، اصلاح کن

---

## 📋 مرحله 3: Full Experiment (10-12 ساعت) ⏱️

### هدف:
تولید نتایج نهایی برای مقاله

### دستور:
```bash
python experiments\focused_reviewer_response_v2.py
```

### چه چیزی اجرا می‌شود:
```
Part 1: Baseline Comparisons (~4 ساعت)
  - OptiGradTrust Full
  - FedAvg Baseline

Part 2: Ablation Study (~6 ساعت)
  - Without Shapley
  - Without VAE
  - With FedAvg (not FedBN-P)

Part 3: Multiple Attacks (~6 ساعت) [OPTIONAL]
  - Scaling attack
  - Label flipping
  - Min-max attack
```

**توجه:** می‌توانید Part 3 را skip کنید (خطوط 378-391 را comment کنید)

### نتیجه مورد انتظار:
```
OptiGradTrust (Full):     ~96.83%
FedAvg Baseline:          ~88.50%
Without Shapley:          ~94.50% (-2.33%)
Without VAE:              ~93.80% (-3.03%)
With FedAvg:              ~91.00% (-5.83%)
```

### خروجی‌ها:
```
experiments/results/focused_reviewer_response/
├── results.json
├── comparison_table.tex
└── ablation_table.tex
```

---

## 📋 مرحله 4: Visualization Update (10 ثانیه) ⏱️

### هدف:
به‌روزرسانی تصاویر با داده‌های واقعی

### دستور:
```bash
python experiments\visualization_suite.py
```

### خروجی:
```
results/aggregated/visualizations/
├── model_rankings.png (updated)
├── detection_metrics_comparison.png (updated)
└── f1_heatmap.png (updated)
```

---

## 📋 مرحله 5: Paper Editing ✍️

حالا آماده‌اید برای ویرایش مقاله!

### چک‌لیست:
- [x] نتایج واقعی Alzheimer
- [x] جداول LaTeX آماده
- [x] تصاویر به‌روز
- [ ] متن مقاله (پروژه دیگر)

### فایل‌های مورد نیاز برای مقاله:
1. `comparison_table.tex` → Section 4 (Results)
2. `ablation_table.tex` → Section 4 (Ablation Study)
3. تصاویر در `results/aggregated/visualizations/`
4. اعداد دقیق از `results.json`

---

## 🚨 نکات بسیار مهم

### ✅ DO:
- ✅ همیشه test های کوچک را اول اجرا کنید
- ✅ نتایج test را بررسی کنید قبل از full run
- ✅ V2 را اجرا کنید (نه V1!)
- ✅ شب قبل از خواب full run را start کنید

### ❌ DON'T:
- ❌ مستقیماً full run را اجرا نکنید
- ❌ V1 را اجرا نکنید (outdated!)
- ❌ نتایج test را نادیده نگیرید
- ❌ terminal را در حین اجرا نبندید

---

## 📊 جدول زمان‌بندی

| مرحله | زمان | شروع توصیه شده | وضعیت |
|-------|------|----------------|--------|
| Quick Test | 10 دقیقه | همین الان | ⏳ |
| Medium Test | 15 دقیقه | بعد از quick | ⏳ |
| Full Run | 10-12 ساعت | شب (قبل خواب) | ⏳ |
| Visualization | 10 ثانیه | صبح (بعد از full) | ⏳ |
| Paper Edit | 2-3 روز | بعد از visualization | ⏳ |

---

## 🎯 دستورالعمل اجرا (گام‌به‌گام)

### امروز (الان):
```bash
# گام 1: Quick test
python experiments\test_quick_verify.py

# اگر موفق شد:
# گام 2: Medium test
python experiments\test_medium_verify.py
```

### امشب (قبل خواب):
```bash
# اگر medium test موفق شد:
# گام 3: Full run
python experiments\focused_reviewer_response_v2.py
```

### فردا صبح:
```bash
# گام 4: Update visualizations
python experiments\visualization_suite.py

# گام 5: بررسی نتایج
dir experiments\results\focused_reviewer_response\
```

---

## ❓ سوالات متداول

### Q: اگر Quick test fail شد چه کنم؟
**A:** Check کنید:
- آیا `config.DATASET = 'alzheimer'` است؟
- آیا دیتاست در `data/alzheimer/` موجود است؟
- لاگ را بخوانید و خطا را پیدا کنید

### Q: اگر Medium test فقط 3/5 pass شد؟
**A:** بستگی دارد کدام check ها fail شدند:
- اگر CHECK 1,2,3 pass شدند: ادامه دهید
- اگر CHECK 4,5 fail شدند: مشکل جدی است، fix کنید

### Q: می‌توانم Part 3 (multiple attacks) را skip کنم؟
**A:** بله! در `focused_reviewer_response_v2.py` خطوط 378-391 را comment کنید.

### Q: چگونه مطمئن شوم V2 اجرا می‌شود نه V1؟
**A:** چک کنید لاگ شامل این است:
```
[Experiment] OptiGradTrust (Full)
...
[1/6] Loading Alzheimer dataset...
```

---

## 🎯 خلاصه نقشه راه

```
Quick Test (10 min)
    ↓ ✅
Medium Test (15 min)
    ↓ ✅
Full Run (10-12 hours)
    ↓ ✅
Update Visualizations (10 sec)
    ↓ ✅
Paper Editing (2-3 days)
    ↓ ✅
Submit to Journal! 🎉
```

---

## 🚀 شروع کنید همین الان!

```bash
cd D:\new_paper
d:/new_paper/venv/Scripts/activate.bat
python experiments\test_quick_verify.py
```

**پس از موفقیت در هر مرحله، به مرحله بعد بروید.**

**موفق باشید!** 🎯✨

