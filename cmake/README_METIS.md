# METIS Integration

This project supports METIS for matrix reordering with flexible configuration options.

## Usage Options

### 1. Automatic (Recommended)
```bash
cmake .. -DUSE_METIS_LIB=ON
```
This will:
- First try to find METIS installed on your system
- If not found, automatically fetch and build METIS v5.2.1 from GitHub

### 2. Force GitHub Version
```bash
cmake .. -DUSE_METIS_LIB=ON -DFORCE_FETCH_METIS=ON
```
Always fetch and build METIS from GitHub, even if system METIS is available.

### 3. System METIS Only
```bash
# Install METIS on your system first
# Arch Linux: sudo pacman -S scotch (provides scotchmetis)
# Ubuntu/Debian: sudo apt install libmetis-dev
# Then:
cmake .. -DUSE_METIS_LIB=ON
```

Set `METIS_ROOT` environment variable to help CMake find your installation:
```bash
METIS_ROOT=/usr/local cmake .. -DUSE_METIS_LIB=ON
```

## Configuration Options

### Data Type Widths
Control METIS index and floating-point precision:

```bash
cmake .. -DUSE_METIS_LIB=ON \
         -DIDXTYPEWIDTH=32 \    # 32 or 64 (default: 32)
         -DREALTYPEWIDTH=64     # 32 or 64 (default: 64)
```

- `IDXTYPEWIDTH`: Width of index type (idx_t)
  - 32: int32_t (suitable for matrices with < 2^31 elements)
  - 64: int64_t (for very large matrices)

- `REALTYPEWIDTH`: Width of floating-point type (real_t)
  - 32: float (single precision)
  - 64: double (double precision, recommended)

## Implementation Details

The METIS setup is handled in `cmake/SetupMETIS.cmake`, which:
1. Checks for system METIS using `FindMETIS.cmake`
2. If not found or `FORCE_FETCH_METIS=ON`, fetches from GitHub
3. Builds GKlib (METIS dependency) first
4. Applies necessary patches for modern CMake compatibility
5. Creates a `METIS::METIS` target for linking

## Troubleshooting

### Build Errors
If you encounter build errors with FetchContent METIS:
- Ensure you have a working C compiler
- Check that CMake version is 3.5 or later
- Try cleaning the build: `rm -rf CMakeCache.txt _deps/`

### System METIS Not Found
If system METIS isn't detected:
- Set `METIS_ROOT`: `cmake .. -DMETIS_ROOT=/path/to/metis`
- Check library is installed: `ldconfig -p | grep metis`
- Use `-DFORCE_FETCH_METIS=ON` to use GitHub version instead

### Link Errors
If you see undefined references to `gk_*` functions:
- This is automatically handled in the FetchContent version
- For system METIS, ensure GKlib is also installed
