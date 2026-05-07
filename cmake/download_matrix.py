import sys
import os
import shutil
import tempfile


def matrix_filename(matrix_name):
    return f"{os.path.basename(os.path.normpath(matrix_name))}.mtx"


def find_matrix_file(path):
    if os.path.isfile(path):
        return path

    for root, _, files in os.walk(path):
        for filename in files:
            if not filename.endswith(".mtx"):
                continue

            candidate = os.path.join(root, filename)
            with open(candidate, "r") as fin:
                headline = fin.readline()
                if "matrix array" in headline:
                    print(f"Skipping {filename} because it contains 'matrix array' in the headline.")
                    continue

            return candidate

    return None

if len(sys.argv) != 3:
    print("Usage: download_matrix.py <matrix_name> <output_dir>")
    sys.exit(1)
matrix_name, outdir = sys.argv[1:]

# normalize path
outdir = os.path.abspath(outdir)
os.makedirs(outdir, exist_ok=True)

outfile = os.path.join(outdir, matrix_filename(matrix_name))

if os.path.exists(outfile):
    print(f"Matrix already exists: {outfile}")
    sys.exit(0)

print(f"Downloading {matrix_name} into {outdir} ...")
try:
    import ssgetpy
except ImportError as exc:
    print("Python package 'ssgetpy' not found. Please install it with: pip install ssgetpy", file=sys.stderr)
    raise SystemExit(1) from exc

matches = ssgetpy.search(matrix_name)
if not matches:
    print(f"No SuiteSparse matrix found for query: {matrix_name}", file=sys.stderr)
    sys.exit(1)

with tempfile.TemporaryDirectory(prefix="ssget-", dir=outdir) as tmpdir:
    path, _ = matches[0].download(destpath=tmpdir, extract=True)
    matrix_path = find_matrix_file(path)
    if matrix_path is None:
        print(f"No Matrix Market coordinate file found for {matrix_name}", file=sys.stderr)
        sys.exit(1)

    tmp_outfile = f"{outfile}.tmp.{os.getpid()}"
    shutil.move(matrix_path, tmp_outfile)
    os.replace(tmp_outfile, outfile)
    print(f"Saved as {outfile}")
