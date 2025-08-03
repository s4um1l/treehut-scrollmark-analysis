# Instagram Scrollmark Analysis POC

This project analyzes Instagram scrollmark data for research purposes.

## Setup

### Backend (Python 3.9)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Install the project with dependencies:
   ```bash
   uv pip install -e .
   ```

4. Install development dependencies (optional):
   ```bash
   uv pip install -e ".[dev]"
   ```

### Development

- Python version: 3.9.6
- Package manager: uv
- Virtual environment: `.venv/`
- Configuration: `pyproject.toml`

### Usage

```bash
cd backend
source .venv/bin/activate
python main.py
```

## Project Structure

```
instagram_scrollmark_analysis_poc/
├── backend/                 # Python backend service
│   ├── main.py             # Main application entry point
│   ├── pyproject.toml      # Project configuration
│   ├── README.md           # Backend-specific documentation
│   └── .venv/              # Virtual environment
├── frontend/               # Frontend application (if needed)
├── .gitignore             # Git ignore rules
└── README.md              # This file
```
