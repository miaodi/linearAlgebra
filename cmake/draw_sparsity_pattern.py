import argparse
import binascii
import math
import os
import struct
import sys
import zlib
from array import array


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_chunk(kind, data):
    checksum = binascii.crc32(kind)
    checksum = binascii.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def write_grayscale_png(filename, width, height, counts, touched_pixels, max_count):
    white_row = b"\x00" + (b"\xff" * width)
    image = bytearray(white_row * height)

    if max_count > 0:
        max_log = math.log1p(max_count)
        stride = width + 1
        for index in touched_pixels:
            count = counts[index]
            density = math.log1p(count) / max_log
            value = int(255.0 * (1.0 - density))
            row = index // width
            col = index - row * width
            image[row * stride + 1 + col] = max(0, min(255, value))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png_data = [
        PNG_SIGNATURE,
        png_chunk(b"IHDR", ihdr),
        png_chunk(b"IDAT", zlib.compress(image, 9)),
        png_chunk(b"IEND", b""),
    ]

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    tmp_filename = f"{filename}.tmp.{os.getpid()}"
    try:
        with open(tmp_filename, "wb") as fout:
            fout.writelines(png_data)
        os.replace(tmp_filename, filename)
    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)


def read_data_line(instream):
    for line in instream:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        return stripped
    return None


def draw_matrix_market(matrix_filename, output_png, max_size):
    if max_size <= 0:
        raise ValueError("max-size must be positive")

    with open(matrix_filename, "r", encoding="utf-8", errors="replace") as fin:
        header = fin.readline().strip().split()
        if len(header) < 5 or header[0].lower() != "%%matrixmarket":
            raise ValueError(f"{matrix_filename} is not a Matrix Market file")

        object_type = header[1].lower()
        storage_format = header[2].lower()
        symmetry = header[4].lower()

        if object_type != "matrix" or storage_format != "coordinate":
            raise ValueError("Only Matrix Market coordinate matrices are supported")

        size_line = read_data_line(fin)
        if size_line is None:
            raise ValueError("Matrix Market file is missing the size line")

        size_fields = size_line.split()
        if len(size_fields) < 3:
            raise ValueError(f"Invalid Matrix Market size line: {size_line}")

        rows, cols, declared_nnz = (int(size_fields[0]), int(size_fields[1]), int(size_fields[2]))
        if rows <= 0 or cols <= 0:
            raise ValueError("Cannot draw an empty matrix")

        image_rows = min(rows, max_size)
        image_cols = min(cols, max_size)
        pixel_count = image_rows * image_cols
        counts = bytearray(pixel_count)
        touched_pixels = array("I")
        max_count = 0

        def mark(row, col):
            nonlocal max_count
            if row < 0 or row >= rows or col < 0 or col >= cols:
                raise ValueError(f"Matrix entry ({row + 1}, {col + 1}) is outside {rows} x {cols}")

            y = row * image_rows // rows
            x = col * image_cols // cols
            index = y * image_cols + x
            count = counts[index]
            if count == 0:
                touched_pixels.append(index)
            if count < 255:
                count += 1
                counts[index] = count
                max_count = max(max_count, count)

        mirror_structure = symmetry in ("symmetric", "hermitian", "skew-symmetric") and rows == cols
        entries_read = 0
        for line in fin:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue

            fields = stripped.split()
            if len(fields) < 2:
                raise ValueError(f"Invalid Matrix Market entry: {stripped}")

            row = int(fields[0]) - 1
            col = int(fields[1]) - 1
            mark(row, col)
            if mirror_structure and row != col:
                mark(col, row)
            entries_read += 1

    if entries_read != declared_nnz:
        print(
            f"Warning: {matrix_filename} declares {declared_nnz} entries but contains {entries_read}",
            file=sys.stderr,
        )

    write_grayscale_png(output_png, image_cols, image_rows, counts, touched_pixels, max_count)
    print(f"Wrote sparsity pattern: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Draw a Matrix Market sparsity pattern as a PNG")
    parser.add_argument("matrix_file", help="Input Matrix Market .mtx file")
    parser.add_argument("output_png", help="Output PNG file")
    parser.add_argument("--max-size", type=int, default=4096, help="Maximum PNG width or height")
    args = parser.parse_args()

    draw_matrix_market(args.matrix_file, args.output_png, args.max_size)


if __name__ == "__main__":
    main()
