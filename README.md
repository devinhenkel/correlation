# Spearman Correlation App (Gradio)

Upload a CSV or XLSX, preview the data, generate a binary copy (> 3 -> 1, else 0), and compute Spearman correlations on either the original numeric data or the binary copy. Download the resulting correlation matrix as CSV.

## Quickstart

1. (Recommended) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open the URL shown in the terminal and upload your file.

## Notes
- Only numeric columns are used for correlation and for generating the binary copy.
- Binary transformation rule: values > 3 are set to 1, all others are set to 0.
- Supported file types: CSV, XLSX.
