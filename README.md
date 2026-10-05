# PCA Face Recognition

Complete academic web application for projections, eigenvectors, PCA,
eigenfaces, and face recognition.

## Team ownership

- Person 1: React frontend
- Person 2: PCA mathematical engine
- Person 3: FastAPI, preprocessing and recognition
- Person 4: experiments, visualization and testing

## Run backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Backend: http://127.0.0.1:8000
Swagger: http://127.0.0.1:8000/docs

## Run frontend

Open another PowerShell:

```powershell
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173

## Generate analysis

```powershell
cd backend
python -m analysis.generate_report
```

## Run tests

```powershell
cd backend
pytest -q
```

The demo uses the Olivetti Faces dataset. For a real deployment, replace the
dataset loader with a properly consented project dataset.
