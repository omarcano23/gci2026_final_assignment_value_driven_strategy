"""Execute the standalone notebook in a fresh temporary directory without altering release files."""
from pathlib import Path
import tempfile
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1]
n=nbformat.read(root/'notebooks/CompanyA_Final_Notebook.ipynb',as_version=4)
with tempfile.TemporaryDirectory() as tmp:
    NotebookClient(n,timeout=120,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
print('PASS: notebook default execution in isolated kernel (no private data).')
