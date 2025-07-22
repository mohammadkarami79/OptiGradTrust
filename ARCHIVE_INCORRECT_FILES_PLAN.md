# 📁 ARCHIVE INCORRECT FILES PLAN

**Purpose:** Systematically archive files containing incorrect 100% CIFAR-10 detection claims

---

## 🚨 **FILES WITH INCORRECT 100% CIFAR-10 CLAIMS**

### **Critical Files Already Corrected (Keep in Place)**
✅ **Corrected - No Archive Needed:**
- `results/final_paper_submission_ready/FINAL_PAPER_TABLE_COMPLETE_IID.md` - CORRECTED ✅
- `results/final_paper_submission_ready/EXECUTIVE_SUMMARY_FOR_ABSTRACT.md` - CORRECTED ✅
- `results/final_paper_submission_ready/COMPLETE_15_ATTACK_SCENARIOS_TABLE.md` - CORRECTED ✅
- `results/final_paper_submission_ready/FINAL_DECISION_FOR_PAPER.md` - CORRECTED ✅
- `results/final_paper_submission_ready/DATA_AUTHENTICITY_VERIFICATION.md` - CORRECTED ✅
- `results/final_paper_submission_ready/COMPLETE_IID_PHASE_RESULTS.md` - CORRECTED ✅
- `results/final_paper_results/FINAL_PAPER_SUMMARY.md` - CORRECTED ✅
- `FINAL_PAPER_WRITING_GUIDE.md` - CORRECTED ✅
- `FINAL_COMPLETE_PAPER_READY_SUMMARY.md` - CORRECTED ✅
- `COMPLETE_IID_vs_NON_IID_COMPARISON.md` - CORRECTED ✅
- `FINAL_SUMMARY_WITH_OPTIMIZATION.md` - CORRECTED ✅

### **Files to Archive (Contain Uncorrected Incorrect Claims)**

#### **Priority 1: High Risk Files**
🔴 **MUST ARCHIVE IMMEDIATELY:**
- `results/final_paper_submission_ready/MEMORY_OPTIMIZED_ANALYSIS_20250629.md`
  - Contains: "CIFAR-10 | ResNet18 | partial_scaling_attack | 62.56% | 100% | 100% | 100% | ✅"
  - Risk: High - in submission folder

#### **Priority 2: Archive Files (May contain other incorrect claims)**
🟡 **RECOMMEND ARCHIVING:**
- Any remaining files in `archive/analysis_files/` containing 100% CIFAR-10 claims
- Any files in `archive/old_results/` with 100% CIFAR-10 claims

---

## 📋 **ARCHIVE PROCEDURE**

### **Step 1: Create Archive Structure**
```
archive/
├── incorrect_results_2025/
│   ├── 100_percent_cifar10_claims/
│   ├── unverified_estimates/
│   └── mixed_authentic_false_data/
```

### **Step 2: Move Files Systematically**
1. **Backup originals** before moving
2. **Document move reasons** in archive
3. **Update any references** in remaining files
4. **Verify no critical data lost**

### **Step 3: Update References**
- Check for any links to archived files
- Update documentation
- Ensure plots/scripts don't reference archived files

---

## ✅ **SAFE TO KEEP (Verified Clean)**

### **Authentic Data Files**
- `results/alzheimer_experiment_summary.txt` ✅
- `VERIFIED_AUTHENTIC_FINAL_RESULTS.md` ✅ 
- `EXPERIMENTAL_RESULTS_AUTHENTICITY_AUDIT.md` ✅

### **Corrected Files (Now Accurate)**
- All files listed in "Critical Files Already Corrected" section above

---

## 🎯 **COMPLETION CRITERIA**

### **Success Indicators:**
✅ No files remain with uncorrected 100% CIFAR-10 claims  
✅ All active files contain only verified authentic data  
✅ Archive properly organized with documentation  
✅ No broken references to archived files  

### **Verification Method:**
```bash
# Search for any remaining 100% CIFAR claims
grep -r "CIFAR.*100%" --exclude-dir=archive/incorrect_results_2025 .
```

---

**Status:** 🟡 **READY TO EXECUTE**  
**Risk Level After Completion:** 🟢 **MINIMAL**  
**Research Integrity:** �� **FULLY RESTORED** 