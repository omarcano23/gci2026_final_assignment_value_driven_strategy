"""Copy only original development-recipe dependencies from the private master archive.
No model fitting, raw data extraction, or reserve evaluation. Never commit the target folder.
"""
from pathlib import Path
import argparse, hashlib, json, zipfile, shutil
p=argparse.ArgumentParser();p.add_argument("master_zip",type=Path);p.add_argument("--work-dir",type=Path,default=Path("private_work"));a=p.parse_args()
root=Path(__file__).resolve().parents[1]; deps=json.loads((root/"model/private_dependencies.json").read_text())
if a.work_dir.exists():raise SystemExit("Choose a new work directory; never overwrite the archive or existing run.")
shutil.copytree(root/"model",a.work_dir)
with zipfile.ZipFile(a.master_zip) as z:
    for rel,expected in deps.items():
        names=[n for n in z.namelist() if n.endswith("/repositorio/"+rel)]
        if len(names)!=1:raise ValueError(f"Expected one exact dependency: {rel}, got {names}")
        b=z.read(names[0]); assert hashlib.sha256(b).hexdigest()==expected, rel
        out=a.work_dir/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b)
print("Private dependency hashes verified. No training or reserve evaluation performed.")
