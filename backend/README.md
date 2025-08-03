# Instagram Scrollmark Analysis - Backend

Backend service for analyzing Instagram scrollmark data using Python 3.9.

## Quick Start

### Prerequisites
- Python 3.9.6
- uv (for fast package management)

### Setup

1. **Create and activate virtual environment:**
   ```bash
   uv venv --python 3.9
   source .venv/bin/activate
   ```

2. **Install the project and dependencies:**
   ```bash
   uv sync
   ```

3. **Install development dependencies (optional):**
   ```bash
   uv sync --extra dev
   ```

4. **Run the application:**
   ```bash
   # Using the main.py entry point
   python main.py
   
   # Or using the installed console script
   instagram-analysis
   ```

## Development

### Project Structure
```
backend/
├── instagram_analysis/     # Main package
│   ├── __init__.py         # Package initialization
│   └── main.py            # Core application logic
├── main.py                # Entry point script
├── pyproject.toml         # Project configuration and dependencies
├── README.md             # This file
└── .venv/                # Virtual environment (auto-created)
```

### Adding Dependencies

Add production dependencies to `pyproject.toml`:
```toml
dependencies = [
    "requests>=2.31.0",
    "pandas>=2.0.0",
]
```

Add development dependencies:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
]
```

Then sync to update dependencies:
```bash
uv sync --extra dev
```

### Code Quality

Format code:
```bash
black .
```

Run linting:
```bash
flake8 .
```

Type checking:
```bash
mypy .
```

Run tests:
```bash
pytest
```

## Features

- Instagram data analysis
- Scrollmark pattern detection
- Data visualization
- Export capabilities

## Configuration

Create a `.env` file for environment variables:
```bash
cp .env.example .env
```

## API Documentation

Once the API is running, documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc` 