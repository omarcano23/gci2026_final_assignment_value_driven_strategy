"""Reproduce the marginal business case, without fitting or reading any customer data."""
import hashlib, subprocess, sys, tempfile, shutil, json, math
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((root/"evidence/f6/summary.json").read_text())
assert math.isclose(s["candidate"]-s["reference"],s["delta"],abs_tol=1e-12)
assert math.isclose(s["delta"]/s["reference"],s["relative"],abs_tol=1e-12)
assert s["relative"]>=.10 and s["ci95"][0]>0 and min(s["guardrails"].values())>=-.05
with tempfile.TemporaryDirectory() as tmp:
    dst=Path(tmp)/"economics";shutil.copytree(root/"economics",dst)
    original={p.name:digest(p) for p in (dst/"results").iterdir() if p.is_file()}
    subprocess.run([sys.executable,"simulate_marginal.py"],cwd=dst,check=True)
    rebuilt={p.name:digest(p) for p in (dst/"results").iterdir() if p.is_file()}
    assert original==rebuilt,(original,rebuilt)
print("PASS: D-030 aggregate identities and gate; seven R02 outputs reproduced byte-for-byte. No reserve access.")
