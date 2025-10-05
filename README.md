
# ECG PDF Cleaner

A simple **Flask web application** that allows you to upload ECG (or any) PDF documents and get their **cleaned version without text**.  
The app removes all text layers from PDFs using PyMuPDF's `redaction` feature while keeping graphical elements (ECG traces, grid, colors) intact.

---

## ✨ Features

- **Upload PDF** → ECG or any PDF containing text layers.  
- **Remove Text** → Deletes all text content, preserves only graphical parts.  
- **Download Clean PDF** → The result is a PDF file with **no selectable/copiable text**, only ECG visuals.

---

## 📦 Requirements

- Python 3.11+
- Flask
- PyMuPDF (fitz)
- Werkzeug
- gunicorn (for deployment on Render)

### requirements.txt

```txt
Flask==3.0.3
PyMuPDF==1.24.9
Werkzeug==3.0.4
gunicorn==23.0.0
```

---

## 🚀 Run Locally

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows → .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
```

Open in browser:
```
http://localhost:5000
```

---

## 🌐 Deploy on Render.com

1. Push code to GitHub.  
2. On Render Dashboard → **Create New Web Service**.  
3. Select repo and branch.  
4. **Build Command:**
   ```bash
   pip install -r requirements.txt
   ```
5. **Start Command:**
   ```bash
   gunicorn app:app --bind 0.0.0.0:$PORT
   ```
6. Deploy → access your app via Render URL.

---

## 📂 Project Structure

```
.
├── app.py              # Main Flask app
├── requirements.txt    # Dependencies
├── README.md           # Project description
├── uploads/            # Uploaded PDFs (created at runtime)
└── outputs/            # Cleaned PDFs (created at runtime)
```

---

## 🔑 Notes

- The app does **not** rasterize pages into images (which may reduce quality).  
- Text is removed via **redaction**:  
  - `page.get_text("words")` finds text,  
  - `page.add_redact_annot()` marks it,  
  - `page.apply_redactions()` deletes it.  
- The result is a PDF where **ECG lines and grid remain**, but **all text is gone**.

---

## 📜 License

MIT License.
