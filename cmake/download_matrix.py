import sys
import os
import ssgetpy
import shutil

if len(sys.argv) != 3:
    print("Usage: download_matrix.py <matrix_name> <output_dir>")
    sys.exit(1)
matrix_name, outdir = sys.argv[1:]

# normalize path
outdir = os.path.abspath(outdir)
os.makedirs(outdir, exist_ok=True)

outfile = os.path.join(outdir, f"{matrix_name}.mtx")

# ✅ check first
if os.path.exists(outfile):
    print(f"Matrix already exists: {outfile}")
    sys.exit(0)

# download
print(f"Downloading {matrix_name} into {outdir} ...")
m = ssgetpy.search(matrix_name)[0]
path,dest = m.download(destpath=outdir, extract=True)

# Check if the downloaded path is a directory
if os.path.isdir(path):
    # If it's a directory, look for .mtx files inside it
    for f in os.listdir(path):
        if f.endswith(".mtx"):
            with open(os.path.join(path, f), "r") as fin:
                headline = fin.readline()
                if "matrix array" in headline:
                    print(f"Skipping {f} because it contains 'matrix array' in the headline.")
                    continue
            # If found, move the .mtx file to the desired output location
            os.rename(os.path.join(path, f), outfile)
            print(f"Saved as {outfile}")
            break
    # remove the directory
    shutil.rmtree(path)
else:
    # If it's not a directory, it's the .mtx file itself, so just rename it
    os.rename(path, outfile)
    print(f"Saved as {outfile}")