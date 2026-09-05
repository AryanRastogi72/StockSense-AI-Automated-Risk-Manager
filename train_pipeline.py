import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "Models"

def run_script(script_name, ticker):
    print(f"\n{'='*50}\nRunning {script_name} for {ticker}\n{'='*50}")
    script_path = MODELS_DIR / script_name
    
    # Run the script and stream output
    process = subprocess.Popen(
        [sys.executable, str(script_path), ticker],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(MODELS_DIR)
    )
    
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    if process.returncode != 0:
        print(f"\nERROR: {script_name} failed with exit code {process.returncode}")
        sys.exit(process.returncode)

def train_pipeline_for_ticker(ticker):
    # 1. Base Models (Untuned)
    for model_type in ["RF", "XG", "LSTM"]:
        for task in ["Class", "Reg"]:
            run_script(f"{model_type}_{task}_Untuned.py", ticker)
            
    # 2. Base Models (Tuned)
    for model_type in ["RF", "XG", "LSTM"]:
        for task in ["Class", "Reg"]:
            run_script(f"{model_type}_{task}_Tuned.py", ticker)
            
    # 3. OOF Generation
    run_script("OOF_Gen_Class.py", ticker)
    run_script("OOF_Gen_Reg.py", ticker)
    
    # 4. Meta-Learner Stacking
    run_script("Stack_Class.py", ticker)
    run_script("Stack_Reg.py", ticker)
    
    # 4. Regime and Risk Calibration
    run_script("Train_Risk_Calibration.py", ticker)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_part_b.py <TICKER>")
        sys.exit(1)
        
    ticker = sys.argv[1]
    train_pipeline_for_ticker(ticker)
