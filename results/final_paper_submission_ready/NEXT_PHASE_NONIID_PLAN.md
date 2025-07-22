# فاز بعدی: آزمایشات Non-IID
## Next Phase: Non-IID Experiments Plan

**📅 Planned Start:** پس از تکمیل CIFAR-10  
**🎯 Goal:** مقایسه IID vs Non-IID attack detection  
**⏱️ Timeline:** 2-3 ساعت برای 3 datasets  

---

## 📊 **فاز 1: IID Results (در حال تکمیل)**

### ✅ **Completed:**
- 🧠 **Alzheimer:** 75% best precision ✅
- 🔢 **MNIST:** 69% best precision ✅  

### 🟡 **In Progress:**
- 🖼️ **CIFAR-10:** Testing... (~20 minutes remaining)

---

## 🚀 **فاز 2: Non-IID Experiments (Next)**

### **Phase 2A: Non-IID Configuration**
```python
NON_IID_CONFIG = {
    'data_distribution': 'non_iid',
    'heterogeneity_level': 0.7,  # 70% non-IID
    'classes_per_client': 2,     # محدود کردن کلاس‌ها
    'alpha_dirichlet': 0.1,      # شدت non-IID
}
```

### **Phase 2B: Datasets & Timeline**
| Dataset | Model | Priority | Est. Time | Expected Challenge |
|---------|-------|----------|-----------|-------------------|
| **MNIST** | CNN | High | 30 min | Easier baseline |
| **Alzheimer** | ResNet18 | High | 45 min | Medical critical |
| **CIFAR-10** | ResNet18 | Medium | 60 min | Most complex |

### **Phase 2C: Research Questions**
1. 📉 **Performance Degradation:** Non-IID vs IID کیفیت
2. 🎯 **Attack Detection:** آیا Non-IID detection را سخت‌تر می‌کند؟
3. 🧠 **Model Robustness:** کدام dataset مقاوم‌تر است؟
4. 📊 **Comparison Analysis:** جدول مقایسه‌ای IID/Non-IID

---

## 📋 **Configuration Strategy:**

### **Memory-Optimized Settings:**
```python
# Base configuration for all Non-IID tests
GLOBAL_EPOCHS = 4         # متوسط برای کیفیت
LOCAL_EPOCHS_CLIENT = 4   
VAE_EPOCHS = 15          # کافی برای detection
BATCH_SIZE = 16          # memory safe
```

### **Priority Attack Types:**
1. 🎯 **Partial Scaling** - معمولا بهترین detection
2. 🔥 **Noise Attack** - قوی در IID
3. 🌪️ **Sign Flipping** - استاندارد

---

## 🎯 **Expected Outcomes:**

### **Hypothesis:**
- 📉 **Detection precision کاهش:** 10-20% در Non-IID
- 📊 **Alzheimer resilient:** medical data مقاوم‌تر
- 🔢 **MNIST moderate impact:** benchmark data
- 🖼️ **CIFAR-10 significant drop:** complex visual data

### **Success Criteria:**
- ✅ **50%+ precision** در Non-IID (vs 60-75% در IID)
- ✅ **Accuracy >90%** حفظ شود
- ✅ **Progressive learning** همچنان مشاهده شود
- ✅ **Reproducible results** با documentation

---

## 📊 **Literature Comparison Ready:**

### **Phase 3: Comparison with 3 Papers**
1. 📄 **FedAvg baseline** (McMahan et al.)
2. 📄 **Byzantine-robust aggregation** (Blanchard et al.)  
3. 📄 **Recent FL attack detection** (2023-2024 papers)

### **Metrics for Comparison:**
- Detection Precision/Recall/F1
- Model Accuracy preservation  
- Computational overhead
- Scalability to client numbers

---

## ⏰ **Timeline Summary:**

```
Phase 1 (IID): [████████████████████▓] 95% (CIFAR-10 finishing)
Phase 2 (Non-IID): [░░░░░░░░░░░░░░░░░░░░] 0% (Ready to start)  
Phase 3 (Comparison): [░░░░░░░░░░░░░░░░░░░░] 0% (Planned)

Total ETA: ~4 hours للمراحل 2-3
```

---

## 🔄 **Ready to Execute:**

**Once CIFAR-10 completes:**
1. ✅ Update complete IID table
2. 🚀 Start Non-IID MNIST (quickest)  
3. 📊 Generate comparison plots
4. 📄 Finalize paper-ready results

**All configurations prepared and tested!** 🎉 