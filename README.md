# Login Anomaly Detection & Risk Scoring System (MLOps-1, 25CSH-320)

A small end-to-end MLOps project. It predicts how risky a login attempt is
(risk score 0-100 → LOW / MEDIUM / HIGH) using a Decision Tree.

## Syllabus mapping
| Syllabus experiment | Where in project |
|---|---|
| Exp 1 – Preprocessing (missing values, scaling, split) | `src/generate_data.py`, `src/train.py` |
| Exp 3 – Decision tree classification + metrics | `src/train.py` |
| Exp 4 – GridSearchCV, tuned vs default | `src/train.py` |
| Exp 5 – Git / DevOps repo | `git init` (steps below) |
| Exp 6 – Docker | `Dockerfile` |
| Exp 7 – Flask REST API | `src/app.py` |
| Exp 8 – CI/CD (GitHub Actions) | `.github/workflows/ci.yml` |
| Unit 3 – Ethics / fairness | fairness check in `train.py` |

## Folder structure
```
data/        generated logins.csv
src/         generate_data.py, train.py, app.py
models/      model.joblib, metrics.json (created by train.py)
tests/       test_model.py
Dockerfile, requirements.txt, .github/workflows/ci.yml
```

## How to run
```bash
pip install -r requirements.txt
python src/generate_data.py     # creates data/logins.csv
python src/train.py             # trains + saves models/model.joblib
pytest -q                       # runs tests
python src/app.py               # starts API on http://localhost:5000
```

Try the API:
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \
 -d '{"hour":3,"failed_attempts":5,"new_device":1,"foreign_country":1,"vpn_used":1,"hours_since_last_login":200}'
```
Response: `{"is_anomaly": true, "risk_level": "HIGH", "risk_score": 100.0}`

## Docker
```bash
docker build -t login-risk .
docker run -p 5000:5000 login-risk
```

## Git + CI/CD (Exp 5 & 8)
```bash
git init && git add . && git commit -m "login risk project"
# push to GitHub -> Actions runs tests + docker build automatically
```

## Features used
`hour`, `failed_attempts`, `new_device`, `foreign_country`, `vpn_used`, `hours_since_last_login`

## Ethics note
`train.py` prints the false-alarm rate for foreign vs local logins. Our model
flags many more genuine foreign logins (~36%) than local ones (~2%) – this is
bias. Real systems should balance classes, review such features, avoid storing
personal data (IP addresses, exact location) and let flagged users verify
themselves instead of being blocked outright.
Note: the data is synthetic (made by a rule + noise), so results are for learning only.

## Results (typical run)
Default tree F1 ≈ 0.54, tuned tree F1 ≈ 0.70, accuracy ≈ 93%.
