"""
standardize_phase07_notebook.py
Cập nhật và thực thi in-place Notebook 07:
paper_v4_10k/notebooks/autonomous_pipeline_loop/07_end_to_end_autonomous_active_loop.ipynb
"""

from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook_07():
    base_dir = Path(__file__).resolve().parent.parent
    nb_path = base_dir / "notebooks" / "autonomous_pipeline_loop" / "07_end_to_end_autonomous_active_loop.ipynb"
    
    print(f"📖 Đang đọc notebook 07 tại: {nb_path}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Cập nhật Cell 8 để hiển thị đầy đủ Bảng Full Benchmark 5 chỉ số cho các vòng và vòng chiến thắng
    cell_8_code = """# Hiển thị Bảng Tiến Hóa 10 Lớp 5 Chỉ Số & Bảng Full Benchmark Vòng Chiến Thắng
summary_md_file = workspace_dir / "results" / "tables" / "per_class_iterations_summary.md"
if summary_md_file.exists():
    with open(summary_md_file, "r", encoding="utf-8") as f:
        display(Markdown(f.read()))

win_iter = verdict.stop_iteration
win_table_file = workspace_dir / "results" / "tables" / f"per_class_active_v{win_iter}.md"
if win_table_file.exists():
    display(Markdown(f"\\n---\\n### 👑 BẢNG FULL BENCHMARK 10 LỚP (5 CHỈ SỐ) CỦA VÒNG CHIẾN THẮNG ACTIVE V{win_iter}"))
    with open(win_table_file, "r", encoding="utf-8") as f:
        display(Markdown(f.read()))
"""
    nb.cells[8].source = cell_8_code

    # Đảm bảo mỗi cell có id
    for idx, cell in enumerate(nb.cells):
        if "id" not in cell:
            cell["id"] = f"cell_{idx:03d}"

    print("🚀 Đang thực thi in-place toàn bộ Notebook 07 qua ExecutePreprocessor...")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})

    print(f"💾 Đang ghi đè in-place lại vào: {nb_path.name}")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    executed_count = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("execution_count") is not None)
    total_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"✅ Hoàn tất Notebook 07! Toàn bộ {executed_count}/{total_code} code cells đã được thực thi và có rich outputs.")

if __name__ == "__main__":
    run_notebook_07()
