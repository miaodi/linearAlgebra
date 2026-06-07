import sys
import os
import shutil
import tarfile
import tempfile
import time


DOWNLOAD_CHUNK_SIZE = 128 * 1024
DOWNLOAD_SLEEP_SECONDS = 0.01


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


def safe_extract(archive, output_dir):
    output_dir = os.path.abspath(output_dir)
    for member in archive.getmembers():
        target = os.path.abspath(os.path.join(output_dir, member.name))
        if not target.startswith(output_dir + os.sep) and target != output_dir:
            raise RuntimeError(f"Archive member escapes output directory: {member.name}")

    archive.extractall(output_dir)


def format_rate(bytes_downloaded, elapsed_seconds):
    if elapsed_seconds <= 0:
        return "unknown speed"

    mib_per_second = bytes_downloaded / (1024 * 1024) / elapsed_seconds
    return f"{mib_per_second:.2f} MiB/s"


def download_matrix_market_archive(matrix, outdir):
    import requests

    local_archive = os.path.join(outdir, f"{matrix.name}.tar.gz")
    bytes_downloaded = 0
    start = time.monotonic()

    response = requests.get(matrix.url("MM"), stream=True)
    response.raise_for_status()

    with open(local_archive, "wb") as fout:
        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
            if not chunk:
                continue

            fout.write(chunk)
            bytes_downloaded += len(chunk)
            time.sleep(DOWNLOAD_SLEEP_SECONDS)

    elapsed = time.monotonic() - start
    print(
        f"Downloaded {bytes_downloaded / (1024 * 1024):.2f} MiB in "
        f"{elapsed:.2f}s ({format_rate(bytes_downloaded, elapsed)})"
    )

    with tarfile.open(local_archive, "r:gz") as archive:
        safe_extract(archive, outdir)

    return outdir

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
    path = download_matrix_market_archive(matches[0], tmpdir)
    matrix_path = find_matrix_file(path)
    if matrix_path is None:
        print(f"No Matrix Market coordinate file found for {matrix_name}", file=sys.stderr)
        sys.exit(1)

    tmp_outfile = f"{outfile}.tmp.{os.getpid()}"
    shutil.move(matrix_path, tmp_outfile)
    os.replace(tmp_outfile, outfile)
    print(f"Saved as {outfile}")
