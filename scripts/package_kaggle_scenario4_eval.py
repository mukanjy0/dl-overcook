"""Package one Scenario 4 evaluator as a self-contained Kaggle CPU kernel."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path


MAIN = '''import base64, json, shutil, subprocess, sys, traceback
from pathlib import Path
P=Path("/kaggle/working/project"); O=Path("/kaggle/working/scenario4"); S=Path("/kaggle/working/run_summary.json")
def save(status, **kw): S.write_text(json.dumps({{"status":status, **kw}}, indent=2))
try:
  save("running", source_commit={commit!r}, config={config!r})
  shutil.unpack_archive(Path("/kaggle/working/project.zip"), P)
  subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check","overcooked-ai==1.1.0","numpy>=1.24,<2","scipy>=1.10,<2","PyYAML>=6.0","Pillow>=10.0"],check=True)
  subprocess.run([sys.executable,"-m","pip","install","--no-deps","-e",str(P)],check=True)
  subprocess.run([sys.executable,str(P/"scripts/evaluate_scenario4_variants.py"),"--config",str(P/{config!r}),"--output-dir",str(O)],check=True,cwd=P)
  save("complete", source_commit={commit!r}, output_dir=str(O), summary=json.loads((O/"variant_summary.json").read_text()))
except Exception as e:
  save("failed", error=repr(e), traceback=traceback.format_exc())
  print(traceback.format_exc(), file=sys.stderr)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--commit", default="HEAD")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", args.commit], cwd=root, text=True).strip()
    config = Path(args.config)
    archive = subprocess.check_output(["git", "archive", "--format=zip", commit, "--", "src", "policies", "configs", "scripts/evaluate_scenario4_variants.py", "pyproject.toml"], cwd=root)
    target = root / "kaggle" / args.version / "input"
    if target.parent.exists(): raise FileExistsError(target.parent)
    target.mkdir(parents=True)
    with zipfile.ZipFile(BytesIO(archive)) as source: source.extractall(target / "project")
    shutil.make_archive(str(target / "project"), "zip", root_dir=target / "project")
    payload = base64.b64encode((target / "project.zip").read_bytes()).decode()
    (target / "main.py").write_text(MAIN.format(commit=commit, config=str(config), payload=payload).replace('Path("/kaggle/working/project.zip")', f'Path("/kaggle/working/project.zip")'), encoding="utf-8")
    text = (target / "main.py").read_text(encoding="utf-8").replace('shutil.unpack_archive(Path("/kaggle/working/project.zip"), P)', f'Path("/kaggle/working/project.zip").write_bytes(base64.b64decode({payload!r})); shutil.unpack_archive(Path("/kaggle/working/project.zip"), P)')
    (target / "main.py").write_text(text, encoding="utf-8")
    (target / "kernel-metadata.json").write_text(json.dumps({"id":args.kernel_id,"title":args.title,"code_file":"main.py","language":"python","kernel_type":"script","is_private":True,"enable_gpu":False,"enable_internet":True,"dataset_sources":[],"competition_sources":[],"kernel_sources":[],"model_sources":[]}, indent=2), encoding="utf-8")
    print(target.parent)


if __name__ == "__main__": main()
