# Data Models & Storage Architecture — EDD v2

## Entity: TrainingImage (Stage 2 — Severity Grader)
- Storage Medium: Local filesystem / Kaggle working directory (image files + folder-based labels)
- Expected Volume: 410 images (Kaggle sheemazain: 214 immature + 196 mature)
- Access Pattern: Read-heavy during training; sequential batch reads; random access for validation splits
- WRITE: Data pipeline only (dataset download/preprocessing)
- READ-ONLY: Training script, evaluation script

### Shape / Structure
```json
{
  "image_id": "string",
  "file_path": "string",
  "modality": "slit-lamp",
  "resolution": { "width": int, "height": int },
  "label": "enum(immature, mature)",
  "source_dataset": "string",
  "license": "string"
}
```

### Actual Split (sheemazain, 80/20 stratified)
| Set | Immature | Mature | Total |
|-----|----------|--------|-------|
| Train | 172 | 157 | 329 |
| Val | 42 | 39 | 81 |
| Total | 214 | 196 | 410 |

## Entity: TrainingImage (Stage 1 — Binary Classifier)
- Storage Medium: Local filesystem (image files + folder-based labels)
- Expected Volume: TBD — needs collection of normal/healthy eye images (~200-400 recommended)
- Access Pattern: Read-heavy during SVM training; sequential batch reads
- WRITE: Data pipeline only (dataset download/preprocessing)
- READ-ONLY: Stage 1 training script (luminance + SVM)

### Shape / Structure
```json
{
  "image_id": "string",
  "file_path": "string",
  "modality": "enum(slit-lamp, external, mobile)",
  "resolution": { "width": int, "height": int },
  "label": "enum(normal, cataract)",
  "source_dataset": "string",
  "license": "string"
}
```

### Note
- Current sheemazain dataset has **no normal class** — all 410 images are cataract (immature/mature)
- Normal images must be collected separately before Stage 1 can be trained

## Entity: ValidationImage (Mobile)
- Storage Medium: Local filesystem (mobile-collected photos)
- Expected Volume: Hundreds to low thousands (collected manually)
- Access Pattern: Write-once during collection; read-only during evaluation
- WRITE: Researcher (manual collection + ophthalmologist labeling)
- READ-ONLY: Evaluation script

### Shape / Structure
```json
{
  "image_id": "string",
  "file_path": "string",
  "device_model": "string",
  "lighting_condition": "string",
  "pupil_dilated": "boolean",
  "ground_truth_label": "enum(normal, immature, mature)",
  "labeler": "string",
  "captured_at": "ISO-8601"
}
```

## Entity: ValidationImage (Mobile)
- Storage Medium: Local filesystem (mobile-collected photos)
- Expected Volume: Hundreds to low thousands (collected manually)
- Access Pattern: Write-once during collection; read-only during evaluation
- WRITE: Researcher (manual collection + ophthalmologist labeling)
- READ-ONLY: Evaluation script

### Shape / Structure
```json
{
  "image_id": "string",
  "file_path": "string",
  "device_model": "string",
  "lighting_condition": "string",
  "pupil_dilated": "boolean",
  "ground_truth_label": "enum(immature, mature)",
  "labeler": "string",
  "captured_at": "ISO-8601"
}
```

## Entity: ModelCheckpoint
- Storage Medium: Kaggle output directory (`.ckpt` / `.h5` / `.keras`)
- Expected Volume: One per checkpoint interval (every 1/5/10 epochs) per model architecture
- Access Pattern: Write-heavy during training; read-only for resumption and export
- WRITE: Training script
- READ-ONLY: Training script (resume), export script

### Shape / Structure
```json
{
  "checkpoint_id": "string",
  "architecture": "string",
  "epoch": int,
  "val_accuracy": float,
  "val_loss": float,
  "file_path": "string",
  "created_at": "ISO-8601"
}
```

## Entity: EvaluationResult
- Storage Medium: Local filesystem (JSON/CSV)
- Expected Volume: One per model checkpoint × two domains (slit-lamp hold-out + mobile validation)
- Access Pattern: Write-once per evaluation; read-only for comparison reporting
- WRITE: Evaluation script
- READ-ONLY: Researcher review, report generation

### Shape / Structure
```json
{
  "model_id": "string",
  "checkpoint_id": "string",
  "domain": "enum(slit-lamp-holdout, mobile-validation)",
  "confusion_matrix": [[int]],
  "accuracy": float,
  "precision_macro": float,
  "recall_macro": float,
  "f1_macro": float,
  "roc_auc_per_class": [float],
  "grad_cam_samples": [string]
}
```

## Entity: InferenceResult
- Storage Medium: Local filesystem / Streamlit app memory (not persisted to server)
- Expected Volume: Ephemeral per inference session
- Access Pattern: Write-once per inference; read-only by UI
- WRITE: Stage 1 SVM + Stage 2 TFLite/SavedModel interpreter
- READ-ONLY: Streamlit UI layer

### Shape / Structure
```json
{
  "input_image_path": "string",
  "preprocessed_shape": [int, int, int],
  "stage1_prediction": "enum(normal, cataract)",
  "stage1_confidence": float,
  "stage2_prediction": "enum(immature, mature, null)",
  "stage2_confidence": "float, null",
  "inference_latency_ms": float
}
```

## Entity: Stage1Model (SVM + Luminance Parameters)
- Storage Medium: Local filesystem (pickle + JSON)
- Expected Volume: One model file + one parameters file
- Access Pattern: Write-once after training; read-only for inference
- WRITE: Stage 1 training script
- READ-ONLY: Streamlit app, evaluation script

### Shape / Structure
```json
{
  "model_type": "svm_gaussian",
  "feature_type": "mean_luminance",
  "preprocessing": ["median_filter", "watershed_segmentation"],
  "decision_threshold": float,
  "healthy_mean_luminance": float,
  "diseased_mean_luminance": float,
  "svm_support_vectors": int,
  "trained_on": "string",
  "created_at": "ISO-8601"
}
```

## Data Trade-offs
- Flat image files + CSV manifest chosen over a database because training pipelines consume filesystem images directly and volume is modest; zero infrastructure cost.
- No normalization of severity labels into separate tables; single flat label per image keeps preprocessing simple.
- Mobile validation set is intentionally isolated (no write access from training code) to prevent leakage.
- Stage 1 (normal vs cataract) and Stage 2 (immature vs mature) use separate datasets and models — Stage 1 needs normal images which are not yet collected.
