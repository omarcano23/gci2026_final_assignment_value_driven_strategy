"""Validate the release files against the immutable content manifest."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'SHA256SUMS.json').read_text())
for name,expected in manifest.items():
    p=root/name
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
print(f'PASS: {len(manifest)} release file hashes.')
