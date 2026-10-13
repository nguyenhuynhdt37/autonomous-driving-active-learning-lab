"""
standardize_phase06_notebook.py
Thực thi in-place Notebook 06 để cập nhật Bảng Full Benchmark 5 chỉ số cho 10 lớp BDD10K.
"""

from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook_06():
    base_dir = Path(__file__).resolve().parent.parent
    nb_path = base_dir / "notebooks" / "06_model_v1_retraining_and_gold_benchmark.ipynb"
    
    print(f"📖 Đang đọc notebook 06 tại: {nb_path}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Đảm bảo mỗi cell có id
    for idx, cell in enumerate(nb.cells):
        if "id" not in cell:
            cell["id"] = f"cell_{idx:03d}"

    print("🚀 Đang thực thi in-place toàn bộ Notebook 06 qua ExecutePreprocessor...")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})

    print(f"💾 Đang ghi đè in-place lại vào: {nb_path.name}")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    executed_count = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("execution_count") is not None)
    total_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"✅ Hoàn tất Notebook 06! Toàn bộ {executed_count}/{total_code} code cells đã được thực thi và có rich outputs.")

if __name__ == "__main__":
    run_notebook_06()
