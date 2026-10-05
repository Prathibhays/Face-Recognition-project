# Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Swagger:
http://127.0.0.1:8000/docs

Generate analysis:

```powershell
python -m analysis.generate_report
```

Tests:

```powershell
pytest -q
```
