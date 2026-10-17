# Experimental Documentation

Tài liệu này mô tả quá trình thực nghiệm theo từng giai đoạn của BDD10K Active Learning Lab.

## Experiment Map

| Phase | Topic | Document | Notebook |
|---|---|---|---|
| 01 | Fast deduplication | [01_fast_dedup.md](01_fast_dedup.md) | `01_dual_stream_fastdedup_benchmark.ipynb` |
| 02 | Data profiling | [02_data_profiling.md](02_data_profiling.md) | `02_automated_data_profiling_benchmark.ipynb` |
| 03 | Frozen dataset splits | [03_frozen_splits.md](03_frozen_splits.md) | `03_shared_frozen_test_val_splits.ipynb` |
| 04 | Seed dataset and baseline | [04_seed_baseline.md](04_seed_baseline.md) | `04_seed_v0_training_ablation.ipynb` |
| 05 | Active mining comparison | [05_active_mining.md](05_active_mining.md) | `05_active_mining_tournament_and_hitl.ipynb` |
| 06 | Model retraining and evaluation | [06_retraining.md](06_retraining.md) | `06_model_v1_retraining_and_gold_benchmark.ipynb` |
| 07 | End-to-end active learning loop | [07_end_to_end_loop.md](07_end_to_end_loop.md) | `07_end_to_end_autonomous_active_loop.ipynb` |
| 08 | Production HITL pipeline | [08_production_hitl.md](08_production_hitl.md) | Verify the current notebook/module path |

## How to Read These Documents

Start with the phase that matches the research question. Each phase document will record the problem, inputs, candidate methods, experimental protocol, metrics, observed results, decision rationale, limitations, and links to reproducible artifacts.

> Status: structure only. Experimental claims and numerical results must be filled from the corresponding notebooks and saved artifacts.
