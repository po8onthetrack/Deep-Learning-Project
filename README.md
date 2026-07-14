# Pose-Based Human Action Recognition for Athletic Movement Analysis

Markerless human action recognition from body pose alone. Given a short sequence of
17 body-keypoint positions tracked across video frames from a single camera, the model
classifies it into one of the 20 [MPOSE2021](https://github.com/PIC4SeR/MPOSE2021_Dataset)
action categories — no motion-capture equipment required.

This is the course project for **APS360 (Applied Fundamentals of Deep Learning)** at the
University of Toronto, by **Harry Zhang**.

---

## ⚠️ Disclosure & Academic Integrity

**This repository is my own individual academic coursework, published for personal
portfolio and reference purposes only.**

- It documents work I completed for a graded University of Toronto course. It is shared
  so others can *read about* the approach and results — **not** as a source to copy from.
- **Do not copy, reuse, adapt, or submit any part of this project — code, text, figures,
  or ideas — as your own work.** If you are currently taking APS360 or a similar course,
  using this material in your own submission is **plagiarism**.
- Plagiarism and unauthorized collaboration are serious academic offenses under the
  University of Toronto's
  [Code of Behaviour on Academic Matters](https://governingcouncil.utoronto.ca/secretariat/policies/code-behaviour-academic-matters-july-1-2019).
  Consequences can include a failing grade, notation on your transcript, suspension, or
  expulsion.
- If you want to reference or build on this work legitimately, please **cite it** (see
  [Citation](#citation) below) and produce your own original implementation.

By viewing this repository you acknowledge that any misuse of its contents is your own
responsibility.

---

## Motivation

The project is motivated by my own experience with badminton and bouldering, where reading
*how* the body moves — a smash, a dynamic reach — is the first step toward better technique
and fewer injuries. Reliable markerless action recognition is a building block for
downstream movement analysis, training feedback, and injury-aware coaching.

Deep learning fits the task because an action is defined by *how* the joints move over
time — a spatio-temporal pattern that cannot be written as a fixed rule over keypoints. A
recurrent network must learn both the pose configuration in each frame and its dynamics
across frames.

## Pipeline

```
 MPOSE2021           Normalize & clean          RNN               Output
 30-frame     ─────► center, scale, mask ─────► over 30    ─────► 20-class probs
 pose seq                    │                   frames             → label
                             │
                             └──────────► MLP baseline ─────────────┘
                                          (averaged pose)
```

At **inference / new-data** time, a frozen pretrained MoveNet CNN converts raw video frames
into the same 17-keypoint COCO format, which is fed through the identical preprocessing
pipeline. MoveNet is never trained.

## Dataset

[**MPOSE2021**](https://github.com/PIC4SeR/MPOSE2021_Dataset) — a dataset for short-time
pose-based human action recognition.

- Each sample is a **30-frame** clip of **17 COCO keypoints**, shape `(30, 17, 3)` as
  `(x, y, confidence)`, with a human-annotated label over **20 action classes** (standing,
  check-watch, sit-down, get-up, walk, wave, box, kick, jog, jump, run, …).
- The **PoseNet** keypoints are used because their 17-joint COCO layout matches the MoveNet
  front-end used at deployment, so no remapping is needed.
- **15,429** clips total. The official cross-subject **split 1** gives **12,562 train** and
  **2,867 test**. A validation set is carved from train by stratified sampling, giving a
  final **train 10,351 / val 2,211 / test 2,867** (≈67 / 14 / 19 %). The held-out test set
  is scored only once.

The dataset is loaded via the [`mpose`](https://pypi.org/project/mpose/) package (see the
notebook); it is not redistributed in this repository.

## Data Processing

Each frame is processed independently and reproducibly:

1. **Normalize** — center the skeleton at the mid-hip and scale by torso length (mid-hip to
   mid-shoulder distance), removing body size, camera distance, and screen position so the
   model sees pose *geometry* rather than raw pixel coordinates.
2. **Clean** — joints with confidence `< 0.2` have their `(x, y, conf)` zeroed together, so
   an undetected/occluded joint is flagged as unreliable rather than entering the network as
   a real position.
3. **Flatten** — each frame becomes a 51-dim vector (`17 × 3`), giving sequences of shape
   `(N, 30, 51)`.

**Class imbalance** is the dominant difficulty (the largest class outnumbers the smallest by
≈9.5×). Inverse-frequency class weights `w_c = N / (K · n_c)` (spanning ≈0.34–3.19) feed a
**weighted cross-entropy** loss, and **macro-F1** is the headline metric so residual neglect
of rare classes stays visible.

## Models

| Model | Description | Params |
|---|---|---|
| Majority-class | Always predict the most frequent action — trivial lower bound | — |
| Averaged-pose MLP (baseline) | 30 frames averaged into one 51-dim vector → `Linear(51,128)→ReLU→Dropout(0.5)→Linear(128,64)→ReLU→Dropout(0.5)→Linear(64,20)` | ≈16 K |
| **Pose RNN / LSTM (primary)** | **2-layer LSTM** (input 51, hidden 128, `batch_first`, inter-layer dropout 0.3); top layer's final hidden state → `Dropout(0.3)` → `Linear(128,20)` | ≈227 K |

The MLP baseline uses the **same keypoint input** as the primary model but discards all
temporal information. Comparing the RNN against it isolates the value of modelling temporal
dynamics: if reading *how* the pose evolves does not beat a single averaged pose, the added
complexity is not justified.

**Training** (identical protocol for a fair comparison): weighted cross-entropy, Adam
(lr `1e-3`), dropout, and early stopping on validation macro-F1. For reproducibility:
seed 66, batch size 64 (train) / 256 (eval), 100-epoch cap, patience 15.

## Results (validation)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority-class (lower bound) | 0.15 | 0.01 |
| Averaged-pose MLP (baseline) | 0.63 | 0.56 |
| **RNN — 2-layer LSTM (primary)** | **0.87** | **0.82** |

The temporal RNN adds **+0.26 macro-F1** over the averaged-pose baseline, confirming the
central hypothesis: modelling *how* the pose evolves is worth the added complexity.
Qualitatively, the RNN resolves precisely the motion-dependent confusions that dominated the
baseline — e.g. the temporal-reverse pair **get-up / sit-down** drops from 0.64 → 0.21, and
the speed pair **run / jog** from 0.41 → 0.11.

**Tuning log** (one change at a time from the 1-layer champion):

| # | Config (change vs. champion) | Val acc | Val macro-F1 | Stop ep. | Verdict |
|---|---|---|---|---|---|
| 0 | 1-layer LSTM, h=128, drop 0.3 (base) | 0.840 | 0.773 | 79 | champion → beaten |
| 1 | `num_layers=2` (+ inter-layer dropout) | 0.871 | **0.819** | 82 | **champion** (+0.046) |
| 2 | + `weight_decay=1e-4` (on 2-layer) | 0.858 | 0.808 | 93 | not adopted (−0.011, within noise) |
| 3 | `hidden=256` (on 2-layer) | 0.858 | 0.804 | 54 | not adopted (−0.015, within noise) |

## New-Data Test Plan

For final testing on truly unseen data: record ≈5 takes each of ≈10–12 feasible actions
(≈50–60 short clips; static camera, full body in frame, position and speed varied), extract
keypoints with frozen MoveNet, apply the identical preprocessing pipeline, and evaluate the
trained model **exactly once** — these clips are never used for training or tuning. The same
setup doubles as a real-time demo.

## Repository Structure

```
.
├── project.ipynb        # main notebook: loading, preprocessing, models, training, eval
├── checkpoints/         # trained weights (.pt)
│   ├── mlp_baseline.pt
│   └── pose_net_baseline.pt
├── data/                # placeholder — dataset loaded via the `mpose` package, not tracked
└── README.md
```

## Setup & Usage

The notebook is written for **Google Colab** (it mounts Google Drive and caches the dataset
there to avoid re-downloading).

1. Open `project.ipynb` in Google Colab (GPU runtime recommended).
2. Run the data-loading cell — it `pip install`s `mpose` and downloads MPOSE2021
   (PoseNet, split 1) the first time, then caches an `.npz` to Drive.
3. Run the preprocessing, split, baseline, and primary-model cells in order.

Core dependencies: `torch`, `numpy`, `scikit-learn`, `matplotlib`, `mpose` (all preinstalled
on Colab except `mpose`).

## Citation

If you reference this project, please cite it:

```bibtex
@misc{zhang2026poseaction,
  author = {Harry Zhang},
  title  = {Pose-Based Human Action Recognition for Athletic Movement Analysis},
  year   = {2026},
  note   = {APS360 course project, University of Toronto},
  howpublished = {\url{https://github.com/po8onthetrack/Deep-Learning-Project}}
}
```

Please also cite the underlying dataset:

> Mazzia, V., Angarano, S., Salvetti, F., Angelini, F., & Chiaberge, M. (2021).
> Action Transformer: A self-attention model for short-time pose-based human action
> recognition. *Pattern Recognition, 124*, 108487.
> ([MPOSE2021 on Zenodo](https://zenodo.org/records/5803076))

## References

- Du, Y., Wang, W., & Wang, L. (2015). Hierarchical recurrent neural network for
  skeleton-based action recognition. *CVPR*, 1110–1118.
- Yan, S., Xiong, Y., & Lin, D. (2018). Spatial temporal graph convolutional networks for
  skeleton-based action recognition. *AAAI*.
- Mazzia, V. et al. (2021). Action Transformer / MPOSE2021. *Pattern Recognition, 124*.
- Bazarevsky, V. et al. (2020). BlazePose: On-device real-time body pose tracking.
  *arXiv:2006.10204*.
- Sun, K., Xiao, B., Liu, D., & Wang, J. (2019). Deep high-resolution representation
  learning for human pose estimation. *CVPR*, 5693–5703.
- [MoveNet: Ultra fast and accurate pose detection model](https://www.tensorflow.org/hub/tutorials/movenet). TensorFlow Hub.

## Acknowledgements

Portions of the writing and code in this project were developed with the assistance of
[Claude](https://claude.ai) and [Claude Code](https://www.anthropic.com/claude-code).

## Ethical Note

The source clips may not represent all body types, ages, or ethnicities, so accuracy may be
uneven across users. A predicted action label can drive downstream training feedback, so a
misclassification could mislead an athlete — the system is intended as a movement-analysis
*aid*, not an authoritative judgement of performance. Pose-based action recognition can also
become surveillance; any deployment that tracks real people must obtain informed consent.
