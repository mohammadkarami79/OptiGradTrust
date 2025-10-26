# شروع کنید - دستورات دقیق

**تاریخ**: 26 اکتبر 2025

---

## ✅ آماده است!

همه چیز implement و test شده:
- ✅ Temperature hybrid در `server.py`
- ✅ Quick test (5 دقیقه)
- ✅ Medium test (40 دقیقه)
- ✅ Full ablation (8 ساعت)

---

## 🚀 دستورات برای اجرا

### گام 1: Quick Test (5 دقیقه)

```bash
cd D:\new_paper
d:/new_paper/venv/Scripts/activate.bat
python experiments\test_temperature_quick.py
```

**انتظار**: پیام "SUCCESS!" + temperature weights نمایش داده شود

---

### گام 2: Medium Test (40 دقیقه)

**فقط اگر گام 1 موفق بود:**

```bash
python experiments\test_temperature_medium.py
```

**انتظار**: Accuracy 85-95% + "Passed: 3/3"

---

### گام 3: Full Ablation (8 ساعت)

**فقط اگر گام 2 موفق بود:**

```bash
python experiments\ablation_temperature_full.py
```

**انتظار**: 5 experiments کامل شوند

---

## 📖 راهنمای کامل

فایل `EXECUTION_GUIDE.md` را بخوانید برای:
- جزئیات هر مرحله
- Troubleshooting
- تحلیل نتایج
- جداول برای paper

---

## ⚠️ مهم

1. **گام‌به‌گام پیش بروید**
2. هر گام را verify کنید قبل از گام بعدی
3. اگر خطا دیدید، متوقف شوید و بگویید
4. Full ablation را شب اجرا کنید (8 ساعت!)

---

## 🎯 هدف

بعد از 3 گام:
- ✅ Temperature hybrid validated
- ✅ Ablation results آماده
- ✅ Paper ready for revision

---

**شروع کنید با Quick Test!** ⚡

```bash
cd D:\new_paper
d:/new_paper/venv/Scripts/activate.bat
python experiments\test_temperature_quick.py
```

