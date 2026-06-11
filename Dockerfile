FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HOME=/opt/doc-worker
ENV XDG_CACHE_HOME=/opt/doc-worker/.cache
ENV RAPIDOCR_MODEL_DIR=/opt/doc-worker/models/RapidOCR
ENV RAPIDOCR_CONFIG=/opt/doc-worker/rapidocr.yaml

RUN apt-get update && apt-get install -y --no-install-recommends \
    inotify-tools \
    ghostscript \
    qpdf \
    unpaper \
    pngquant \
    jbig2dec \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    ocrmypdf \
    ocrmypdf-rapidocr \
    modelscope \
    pyyaml \
    requests

RUN python - <<'PY'
import os
from pathlib import Path
from modelscope import snapshot_download

target = Path(os.environ["RAPIDOCR_MODEL_DIR"])
target.mkdir(parents=True, exist_ok=True)

print(f"Downloading RapidOCR models into {target}")
snapshot_download(
    repo_id="RapidAI/RapidOCR",
    local_dir=str(target),
)
print("RapidOCR model download complete.")

print("Available ONNX models:")
for p in sorted(target.rglob("*.onnx")):
    print(p)
PY

# Build a config file pointing RapidOCR at local models.
# The exact filenames can vary between RapidOCR releases, so this discovers them.
RUN python - <<'PY'
import os
from pathlib import Path
import shutil
import yaml
import rapidocr

model_root = Path(os.environ["RAPIDOCR_MODEL_DIR"])
config_path = Path(os.environ["RAPIDOCR_CONFIG"])

onnx_files = list(model_root.rglob("*.onnx"))

def find_one(*needles):
    matches = []
    for p in onnx_files:
        full = str(p).lower()
        if all(n.lower() in full for n in needles):
            matches.append(p)

    if not matches:
        raise FileNotFoundError(
            f"No ONNX model found for needles={needles}\n\nAvailable:\n"
            + "\n".join(str(p) for p in sorted(onnx_files))
        )

    matches = sorted(
        matches,
        key=lambda p: (
            "infer" not in p.name.lower(),
            "mobile" not in p.name.lower(),
            len(str(p)),
            str(p),
        ),
    )
    return matches[0]

def find_dict(*needles):
    candidates = []
    for p in model_root.rglob("*"):
        if not p.is_file():
            continue
        full = str(p).lower()
        name = p.name.lower()
        if all(n.lower() in full or n.lower() in name for n in needles):
            if p.suffix.lower() in [".txt", ".dict"]:
                candidates.append(p)

    if not candidates:
        return None

    return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]

def find_default_rapidocr_config():
    pkg_root = Path(rapidocr.__file__).resolve().parent
    candidates = list(pkg_root.rglob("*.yaml")) + list(pkg_root.rglob("*.yml"))

    valid = []

    for candidate in candidates:
        try:
            data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        if all(k in data for k in ["Det", "Cls", "Rec"]):
            det = data.get("Det") or {}
            cls = data.get("Cls") or {}
            rec = data.get("Rec") or {}

            if isinstance(det, dict) and isinstance(cls, dict) and isinstance(rec, dict):
                if "engine_type" in det and "engine_type" in cls and "engine_type" in rec:
                    valid.append(candidate)

    if not valid:
        raise FileNotFoundError(
            "Could not find RapidOCR default config with Det/Cls/Rec engine_type.\n"
            "Available YAML files:\n"
            + "\n".join(str(p) for p in candidates)
        )

    # Prefer the most generic default config.
    valid = sorted(
        valid,
        key=lambda p: (
            "default" not in p.name.lower(),
            len(str(p)),
            str(p),
        ),
    )
    return valid[0]

det_model = find_one("multi", "det")
rec_model = find_one("latin", "rec")
cls_model = find_one("cls")

latin_dict = (
    find_dict("latin")
    or find_dict("ppocr", "keys")
    or find_dict("dict")
)

default_config = find_default_rapidocr_config()

print(f"Using RapidOCR default config: {default_config}")
print(f"Using det model: {det_model}")
print(f"Using rec model: {rec_model}")
print(f"Using cls model: {cls_model}")
print(f"Using rec dict: {latin_dict}")

cfg = yaml.safe_load(default_config.read_text(encoding="utf-8"))

# Patch only the local model paths. Keep all required RapidOCR schema keys.
cfg["Det"]["model_path"] = str(det_model)
cfg["Cls"]["model_path"] = str(cls_model)
cfg["Rec"]["model_path"] = str(rec_model)

# Some RapidOCR versions require a recognition dictionary path.
# Keep existing default if no Latin dict was found.
if latin_dict is not None:
    for key in ["dict_path", "rec_keys_path", "character_dict_path"]:
        if key in cfg["Rec"]:
            cfg["Rec"][key] = str(latin_dict)

# Be explicit. RapidOCR's enum conversion expects valid engine_type values.
for section in ["Det", "Cls", "Rec"]:
    cfg[section]["engine_type"] = "onnxruntime"

config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

print(f"Wrote {config_path}:")
print(config_path.read_text())
PY

# Verify plugin and option exist during build.
RUN ocrmypdf --plugin ocrmypdf_rapidocr --help | grep -E "rapidocr|plugin" >/dev/null

COPY worker.py /usr/local/bin/worker.py
RUN chmod +x /usr/local/bin/worker.py

ENTRYPOINT ["python", "/usr/local/bin/worker.py"]
