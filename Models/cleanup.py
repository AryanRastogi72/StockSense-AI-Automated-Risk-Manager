import os
import json
import glob

scripts = [
    "LSTM_Class_Tuned.py", "LSTM_Class_Untuned.py",
    "LSTM_Reg_Tuned.py", "LSTM_Reg_Untuned.py",
    "RF_Class_Tuned.py", "RF_Class_Untuned.py",
    "RF_Reg_Tuned.py", "RF_Reg_Untuned.py",
    "XG_Class_Tuned.py", "XG_Class_Untuned.py",
    "XG_Reg_Tuned.py", "XG_Reg_Untuned.py"
]

tuned_tree_scripts = [
    "RF_Class_Tuned.py",
    "RF_Reg_Tuned.py",
    "XG_Class_Tuned.py",
    "XG_Reg_Tuned.py"
]

for script in scripts:
    with open(script, "r") as f:
        content = f.read()

    if "if __name__ ==" in content:
        print(f"Skipping {script}, already has main block.")
        continue

    # Add json import if not present
    if "import json" not in content:
        content = "import json\n" + content

    # Find the split point
    if "torch.manual_seed(" in content:
        split_idx = content.find("torch.manual_seed(")
    elif "df = pd.read_csv(" in content:
        split_idx = content.find("df = pd.read_csv(")
    else:
        print(f"Could not find split point in {script}")
        continue

    # Add json saving for tuned tree scripts
    if script in tuned_tree_scripts:
        if "Class" in script:
            target_str = 'f1_baseline = f1_score(y_test, y_pred_baseline, average="macro")\n'
            variant_dir = script.replace(".py", "").lower()
            model_name = script.split("_")[0]
            metrics_code = f"""
baseline_metrics = {{
    "accuracy_naive": float(accuracy_naive),
    "accuracy_baseline": float(accuracy_baseline),
    "precision_baseline": float(precision_baseline),
    "recall_baseline": float(recall_baseline),
    "f1_baseline": float(f1_baseline)
}}
from pathlib import Path
save_path = Path("saved_models") / TICKER / "{variant_dir}"
save_path.mkdir(parents=True, exist_ok=True)
with open(save_path / f"{{TICKER}}_{model_name}_baseline_comparison.json", "w") as f:
    json.dump(baseline_metrics, f, indent=4)
"""
            content = content.replace(target_str, target_str + metrics_code)
        else:
            target_str = 'r2_naive = r2_score(y_true_price, y_naive_price)\n'
            variant_dir = script.replace(".py", "").lower()
            model_name = script.split("_")[0]
            metrics_code = f"""
baseline_metrics = {{
    "rmse_naive": float(rmse_naive),
    "rmse_baseline": float(rmse_baseline),
    "r2_naive": float(r2_naive),
    "r2_baseline": float(r2_baseline)
}}
from pathlib import Path
save_path = Path("saved_models") / TICKER / "{variant_dir}"
save_path.mkdir(parents=True, exist_ok=True)
with open(save_path / f"{{TICKER}}_{model_name}_baseline_comparison.json", "w") as f:
    json.dump(baseline_metrics, f, indent=4)
"""
            content = content.replace(target_str, target_str + metrics_code)

    # Now split and indent
    top_part = content[:split_idx]
    bottom_part = content[split_idx:]

    indented_bottom = "\n".join(["    " + line if line else line for line in bottom_part.split("\n")])

    final_content = top_part.rstrip() + "\n\nif __name__ == '__main__':\n" + indented_bottom + "\n"

    with open(script, "w") as f:
        f.write(final_content)
    print(f"Processed {script}")
