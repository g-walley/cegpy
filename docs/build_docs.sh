# Install required Python dependencies (Sphinx etc.)
python3 -m pip install -r docs/requirements.txt

# Run Jupyter Book
jupyter-book build docs/
