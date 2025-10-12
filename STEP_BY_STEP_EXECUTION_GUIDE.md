# راهنمای گام‌به‌گام اجرای کامل آزمایش‌ها

## 🎯 هدف
تولید **تمام نتایج و تصاویر مورد نیاز** برای پاسخ به داوران و تکمیل مقاله

---

## 📊 مراحل اجرا (به ترتیب اولویت)

### ⭐ مرحله 1: Focused Reviewer Response (اولویت بالا)
**این را اول اجرا کنید!**

```bash
cd D:\new_paper
d:/new_paper/venv/Scripts/activate.bat
python experiments\focused_reviewer_response.py
```

**زمان:** 8-10 ساعت  
**چرا اول؟** مستقیماً نیازهای داوران را برطرف می‌کند:
- ✅ مقایسه با baseline‌ها (FLGuard, FLTrust)
- ✅ Ablation study (Shapley, VAE, FedBN-P)
- ✅ جداول LaTeX آماده برای مقاله

**خروجی:**
```
experiments/results/focused_reviewer_response/
├── comparison_table.tex
├── ablation_table.tex
└── results.json
```

**بعد از اجرا:**
- ✅ جداول را در مقاله قرار دهید
- ✅ نتایج را بررسی کنید
- ✅ به مرحله بعد بروید

---

### 🔬 مرحله 2: Visualization Suite (برای تصاویر گمشده)
**این را بعد از مرحله 1 اجرا کنید**

```bash
python experiments\visualization_suite.py
```

**زمان:** 30-60 دقیقه  
**چرا؟** تصاویر مورد نیاز مقاله را تولید می‌کند:
- ✅ `model_rankings.png`
- ✅ `detection_metrics_comparison.png`
- ✅ `f1_heatmap.png`

**خروجی:**
```
results/aggregated/visualizations/
├── model_rankings.png
├── detection_metrics_comparison.png
├── f1_heatmap.png
└── ... (و تصاویر دیگر)
```

**نکته مهم:** این script از نتایج مرحله 1 استفاده می‌کند، پس باید بعد از آن اجرا شود.

---

### 📈 مرحله 3: Extended Metrics (برای نتایج جامع‌تر)
**اختیاری اما توصیه می‌شود**

```bash
python experiments\extended_metrics.py
```

**زمان:** 4-6 ساعت  
**چرا؟** معیارهای اضافی مورد نیاز medical diagnosis:
- ✅ Precision, Recall, F1-score
- ✅ AUC-ROC
- ✅ Confusion matrices
- ✅ Per-class metrics

**خروجی:**
```
experiments/results/extended_metrics/
├── confusion_matrices/
├── roc_curves/
├── per_class_metrics.csv
└── extended_metrics_table.tex
```

---

### 🔥 مرحله 4: Confidence Intervals (برای اعتبار آماری)
**اختیاری اما قوی می‌کند مقاله را**

```bash
python experiments\confidence_intervals.py
```

**زمان:** 20-30 ساعت (چند seed مختلف)  
**چرا؟** اعتبار آماری به نتایج می‌دهد:
- ✅ Multiple random seeds
- ✅ Mean ± confidence interval
- ✅ Statistical significance tests

**خروجی:**
```
experiments/results/confidence_intervals/
├── confidence_intervals_table.tex
├── statistical_tests.json
└── seeds_comparison.png
```

**نکته:** این را می‌توانید موازی با سایر کارها در background اجرا کنید.

---

### ⚡ مرحله 5: Computational Overhead (برای داوران)
**توصیه می‌شود**

```bash
python experiments\computational_overhead.py
```

**زمان:** 2-3 ساعت  
**چرا؟** داوران پرسیدند overhead چقدر است:
- ✅ Runtime profiling
- ✅ Memory usage
- ✅ Per-component cost

**خروجی:**
```
experiments/results/computational_overhead/
├── overhead_comparison_table.tex
├── runtime_breakdown.png
└── memory_usage.png
```

---

### 🔢 مرحله 6: Scalability Tests (اختیاری)
**فقط اگر زمان دارید**

```bash
python experiments\scalability_tests.py
```

**زمان:** 15-20 ساعت  
**چرا؟** نشان می‌دهد سیستم برای تعداد زیاد client مقیاس‌پذیر است:
- ✅ 10, 20, 50, 100 clients
- ✅ Different malicious ratios

**خروجی:**
```
experiments/results/scalability/
├── scalability_table.tex
└── scalability_curves.png
```

---

## 🎯 توصیه من (برای شما)

با توجه به زمان محدود و نیاز به پاسخ سریع به داوران:

### اجرای اولویت بالا (48 ساعت)
```bash
# روز 1 (شب تا صبح)
python experiments\focused_reviewer_response.py

# روز 1 (بعد از ظهر)
python experiments\visualization_suite.py

# روز 2 (شب تا صبح)
python experiments\extended_metrics.py
python experiments\computational_overhead.py
```

این 3 مرحله **تمام نیازهای اصلی داوران** را برطرف می‌کند:
- ✅ Baseline comparison
- ✅ Ablation study
- ✅ Extended metrics
- ✅ Computational overhead
- ✅ تمام تصاویر مورد نیاز

---

## 🔧 حل مشکل تصاویر گمشده

شما این خطاها را دارید:
```
file not found: ../results/aggregated/visualizations/model_rankings.png
file not found: ../results/aggregated/visualizations/detection_metrics_comparison.png
file not found: ../results/aggregated/visualizations/f1_heatmap.png
```

### راه‌حل:

#### گزینه 1: اجرای Visualization Suite (توصیه می‌شود)
```bash
python experiments\visualization_suite.py
```
این تمام تصاویر را از نتایج موجود تولید می‌کند.

#### گزینه 2: اگر نتایج ندارید
ابتدا `focused_reviewer_response.py` را اجرا کنید، سپس `visualization_suite.py`.

#### گزینه 3: موقت (برای کامپایل LaTeX)
```latex
% در main.tex این خطوط را موقتاً comment کنید:
% \includegraphics{../results/aggregated/visualizations/model_rankings.png}
% \includegraphics{../results/aggregated/visualizations/detection_metrics_comparison.png}
% \includegraphics{../results/aggregated/visualizations/f1_heatmap.png}
```

---

## 📋 چک‌لیست اجرا

### مرحله 1: آماده‌سازی
- [ ] venv فعال است
- [ ] GPU در دسترس است
- [ ] دیتاست Alzheimer موجود است
- [ ] حداقل 50GB فضای خالی روی دیسک

### مرحله 2: اجرای اصلی (اولویت 1)
- [ ] `focused_reviewer_response.py` ✅
- [ ] نتایج را بررسی کردم
- [ ] جداول LaTeX تولید شد

### مرحله 3: تصاویر
- [ ] `visualization_suite.py` ✅
- [ ] تصاویر در مسیر صحیح قرار دارند
- [ ] LaTeX بدون خطا کامپایل می‌شود

### مرحله 4: نتایج اضافی (اختیاری)
- [ ] `extended_metrics.py` ✅
- [ ] `computational_overhead.py` ✅
- [ ] `confidence_intervals.py` (در صورت داشتن زمان)

---

## ⚠️ نکات مهم

### 1. ترتیب اجرا مهم است!
```
focused_reviewer_response → visualization_suite → extended_metrics
```
چون هر کدام از نتایج قبلی استفاده می‌کنند.

### 2. اجرای موازی
می‌توانید این‌ها را همزمان اجرا کنید (در terminal‌های مختلف):
```bash
# Terminal 1
python experiments\focused_reviewer_response.py

# Terminal 2 (بعد از 2-3 ساعت)
python experiments\computational_overhead.py
```

### 3. Background execution
برای اجراهای طولانی:
```bash
start /B python experiments\confidence_intervals.py > confidence_intervals.log 2>&1
```

### 4. مانیتورینگ
```bash
# برای دیدن progress
tail -f experiments/results/*/log.txt

# برای چک کردن GPU
nvidia-smi -l 1
```

---

## 🎯 خلاصه برای شما

### همین الان:
```bash
cd D:\new_paper
d:/new_paper/venv/Scripts/activate.bat
python experiments\focused_reviewer_response.py
```

### 8-10 ساعت بعد:
```bash
python experiments\visualization_suite.py
```

### بررسی نتایج:
```bash
# جداول LaTeX
notepad experiments\results\focused_reviewer_response\comparison_table.tex
notepad experiments\results\focused_reviewer_response\ablation_table.tex

# تصاویر
explorer results\aggregated\visualizations\
```

### قرار دادن در مقاله:
1. جداول را در Section 4 (Results) قرار دهید
2. تصاویر را در figures/ کپی کنید
3. LaTeX را کامپایل کنید
4. ✅ Done!

---

## 📞 در صورت بروز مشکل

### خطای GPU:
```python
# در config.py
USE_CUDA = False  # موقتاً GPU را غیرفعال کنید
```

### خطای Memory:
```python
# در config.py
BATCH_SIZE = 16  # کاهش batch size
NUM_CLIENTS = 5  # کاهش موقت تعداد client
```

### خطای Dataset:
```bash
# بررسی وجود فایل‌ها
dir D:\new_paper\data\alzheimer\train
dir D:\new_paper\data\alzheimer\test
```

### Script متوقف می‌شود:
```bash
# اضافه کردن exception handling
python -u experiments\focused_reviewer_response.py 2>&1 | tee output.log
```

---

## ✅ بعد از تکمیل همه مراحل

شما خواهید داشت:
- ✅ جداول مقایسه با baseline‌ها
- ✅ جداول ablation study
- ✅ تمام تصاویر مورد نیاز
- ✅ معیارهای extended
- ✅ تحلیل computational overhead
- ✅ اعتبار آماری (confidence intervals)

این **تمام چیزی است که برای پاسخ به داوران** نیاز دارید! 🎉

---

**الان شروع کنید! زمان طلاست!** ⏱️🚀

