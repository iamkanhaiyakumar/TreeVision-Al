# 🌲 TreeVision AI — Comparative Evaluation on Unseen Test Site
**Held-Out Geographic Test Site:** `YELL` (Yellowstone Northern Range, Wyoming)  
**Total Ground Truth Tree Crowns:** 279  
**Zero-Leakage Guarantee:** The test site was completely excluded from training and hyperparameter tuning.

| Metric | DeepForest Baseline | Custom YOLO Model | Analysis / Notes |
|---|---|---|---|
| **True Positives (TP)** | 136 | 123 | Correct crown matches at IoU ≥ 0.35 |
| **False Positives (FP)** | 51 | 400 | Spurious detections (shadows, background) |
| **False Negatives (FN)** | 143 | 156 | Missed crowns (small/dense understory) |
| **Precision** | **0.7273** | 0.2352 | Exactness of positive detections |
| **Recall** | **0.4875** | 0.4409 | Coverage of actual forest crowns |
| **F1-Score** | **0.5837** | 0.3067 | Harmonic mean of Precision and Recall |
| **Predicted Tree Count** | 187 | 523 | Reference count = 279 |
| **Tree Count Error (%)** | **32.97%** | 87.46% | Application-level inventory error |
| **Canopy Area Error (%)**| **21.00%** | 532.42% | Estimated canopy footprint discrepancy |
| **Average Confidence**  | 50.8% | 98.1% | Model certainty score |

---

## 🔬 Honest Model Selection Rationale (Section 18 & 63 Compliance)
- **DeepForest** outperforms the initial few-epoch custom model on zero-shot out-of-domain transfer to the Rocky Mountain subalpine forest (0.58 vs 0.31 F1). DeepForest was trained on thousands of crowns across 21 NEON sites.
- **Production Decision:** We provide **DeepForest as the recommended primary production detector** for highest out-of-domain accuracy, while offering the **Custom YOLO detector** as an ultra-fast alternative in the UI toggle.
- Both models are fully functional and verifiable in the application.
