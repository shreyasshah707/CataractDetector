# Architecture Overview — EDD v2

## Modules & Responsibilities
- `Kaggle Training Pipeline` (Python notebooks): Owns dataset download, preprocessing, model training, checkpointing, evaluation, and model export for Stage 2 (EfficientNet-B0 severity grader). Does not touch mobile validation photos directly.
- `Stage 1 Binary Classifier` (Luminance + SVM): Owns normal vs cataract detection. Implements paper's luminance feature extraction (median filter + watershed + mean luminance) + SVM with Gaussian kernel. Lightweight, no GPU needed.
- `Stage 2 Severity Grader` (EfficientNet-B0): Owns immature vs mature classification for diseased eyes only. Receives input only when Stage 1 detects cataract.
- `Mobile Validation Set` (collected photos + ophthalmologist labels): Owns ground-truth labels for mobile-captured eye photos. Write-only during collection; read-only during evaluation. Does not influence training.
- `Exported Model` (TensorFlow SavedModel + ONNX): Owns the trained Stage 2 model in research-friendly formats. Read-only by evaluation scripts.
- `Stage 1 Model` (SVM pickle + luminance parameters): Owns the trained Stage 1 binary classifier. Read-only by evaluation scripts.

## Trust Boundaries & Data Entry
- Untrusted Input Points:
  - Mobile camera/gallery image files (arbitrary resolution, lighting, quality) — used only for validation, never training.
  - Public training dataset labels (trusted source but potentially noisy severity labels).
- Internal Boundaries:
  - Stage 1 (Luminance + SVM) -> binary decision (normal/cataract)
  - Stage 2 (EfficientNet-B0) -> severity grade (immature/mature) — only invoked if Stage 1 = cataract
  - Training pipeline (trusted, local/Kaggle) -> exported SavedModel/ONNX -> Evaluation script (research-only).

## System Map
```mermaid
flowchart LR
    A[Public Slit-Lamp Dataset] -->|Download + Preprocess| B(Kaggle Training Pipeline - Stage 2)
    B -->|Train + Checkpoints| B
    B -->|Export SavedModel + ONNX| C[Stage 2 Model Files]
    D[Normal + Cataract Images] -->|Luminance features + SVM| E[Stage 1 Binary Classifier]
    E -->|Binary decision| F{Normal or Cataract?}
    F -->|Normal| G[Output: No Cataract Detected]
    F -->|Cataract| C
    C -->|Severity grade| H[Output: Immature or Mature]
    I[Mobile Validation Photos] -->|Ophthalmologist labels| J[Validation Ground Truth]
    J -->|Evaluate against| B
    J -->|Evaluate against| E
    C -->|Load for inference| K[Research Evaluation Script]
    E -->|Load for inference| K
    K -->|Metrics + Domain Gap Report| L[Researcher Review]
```

## Links
- `docs/DECISIONS.md` — Recorded architectural decisions (D-001 through D-009).
- `docs/FLOW.md` — Step-by-step execution traces for training and evaluation.
- `docs/DATA.md` — Data schemas and access patterns.
- `docs/PLAN.md` — Overall project plan, timeline, and risks.
