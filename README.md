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
pip install -r requirements.txt
```

### 4. Run baselines
```powershell
python -m Runners.run_baselines
```
