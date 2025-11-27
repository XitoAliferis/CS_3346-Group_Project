# CS_3346-Group_Project

## How to Run

### 1. Create a virtual environment
```powershell
python -m venv venv
```

### 2. Activate the virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3. Install required packages
```powershell
pip install "numpy<2"
pip install transformers==4.57.3
pip install peft optuna
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install datasets
pip install evaluate
```

### 4. Run baselines
```powershell
python -m Runners.run_baselines
```
