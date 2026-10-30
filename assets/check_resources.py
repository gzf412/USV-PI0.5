"""Check packaged SDF/COLLADA resource references without a running simulator."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

ASSETS = Path(__file__).resolve().parent
MODELS = ASSETS / "models"

def check():
    checked = 0
    failures = []
    for filename in sorted(ASSETS.rglob("*")):
        if filename.suffix not in (".sdf", ".dae"):
            continue
        root = ET.parse(filename).getroot()
        for node in root.iter():
            tag = node.tag.split("}")[-1]
            # COLLADA init_from in library_images contains file paths; other
            # init_from nodes refer to internal shader sampler IDs.
            if tag == "image":
                nodes = [n for n in node if n.tag.split("}")[-1] == "init_from"]
            elif tag in ("uri", "albedo_map", "roughness_map", "normal_map", "metalness_map", "vertex", "fragment"):
                nodes = [node]
            else:
                continue
            for ref in nodes:
                value = (ref.text or "").strip()
                if not value:
                    continue
                checked += 1
                path = MODELS / value[len("model://"):] if value.startswith("model://") else filename.parent / value
                if value.startswith(("/", "http://", "https://", "file://")):
                    failures.append({"file": str(filename.relative_to(ASSETS)), "reference": value, "reason": "nonportable or remote reference"})
                elif not path.exists():
                    failures.append({"file": str(filename.relative_to(ASSETS)), "reference": value, "reason": "missing resource"})
    return {"references_checked": checked, "failures": failures}

if __name__ == "__main__":
    result = check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result["failures"]))
