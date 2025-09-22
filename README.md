# Resume Tailor - AI Candidate Matching

An interactive Streamlit app that analyzes a Job Description and matches uploaded CVs/resumes using an LLM. It extracts candidate data, scores candidates against the role, and lets you export the results.

## Features
- Paste a Job Description and get a structured breakdown
- Upload multiple CVs (PDF/DOCX/TXT) and auto-extract details
- Advanced AI matching with per-dimension scores
- Visual dashboards and per-candidate analysis
- Export results to CSV

## Requirements
- Python 3.10+
- An OpenAI API key with access to a chat model

## Quickstart
1) Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) Configure environment variables:
- Copy `.env.example` to `.env`
- Put your OpenAI key in the `.env` file:

```
OPENAI_API_KEY=sk-...
```

4) Run the app:

```powershell
streamlit run main.py
```

The app will open in your browser. Start by analyzing a Job Description, then upload CVs, and finally run the analysis and export results.

## Configuration notes
- The model is set in `main.py` via `ChatOpenAI(model="gpt-5-mini", temperature=1)`. If your account doesn’t have that model, change it to a model you can access (e.g. `gpt-4o-mini`).
- Supported CV formats: PDF, DOCX, TXT

## Project structure
- `main.py` — Streamlit app entry point used in README and recommended for running
- `app.py` — Present in the repo (not wired in this README)

## Troubleshooting
- If Streamlit doesn’t start, ensure your virtual environment is active and packages are installed.
- If model calls fail, verify `OPENAI_API_KEY` is set and your model name is valid for your account.

## License
Add a license file if you plan to share this publicly.
