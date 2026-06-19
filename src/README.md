# Financial Transaction Monitoring System

An AI-powered API that monitors financial transactions for fraud and high-risk activities using a specialized FinBERT model from Hugging Face.

## Installation

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv testenv
   source testenv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the API

```bash
uvicorn src.main:app --reload
```
Open `http://127.0.0` in your browser to test the endpoints.
