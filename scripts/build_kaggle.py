"""Build self-contained Kaggle kernel scripts: embeds src/gdsl1 into run.py
(Kaggle script kernels are a single file; this avoids needing GitHub access)."""
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sources():
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "src", "gdsl1", "**", "*.py"), recursive=True)):
        rel = os.path.relpath(p, os.path.join(ROOT, "src")).replace(os.sep, "/")
        with open(p, encoding="utf-8") as f:
            out[rel] = f.read()
    return out


def main():
    for tpl in glob.glob(os.path.join(ROOT, "kaggle", "*", "run_template.py")):
        with open(tpl, encoding="utf-8") as f:
            text = f.read()
        text = text.replace("SOURCES = {}  # @@SOURCES@@", "SOURCES = " + repr(sources()))
        out = os.path.join(os.path.dirname(tpl), "run.py")
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("wrote", out, len(text), "bytes")


if __name__ == "__main__":
    main()
