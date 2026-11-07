#!/bin/bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out=$(mkdir -p "$1" && cd "$1" && pwd); shift
gen=$out/pgo-gen; prof=$out/pgo-profile
rm -rf "$gen"
export CCACHE_DISABLE=1
cmake -S "$root" -B "$gen" "$@" -DCMAKE_BUILD_TYPE=Release -DROCKETGPU_BUILD_PYTHON=ON \
    -DROCKETGPU_PGO=GENERATE -DROCKETGPU_PGO_PROFILE="$prof"
{ IFS= read -r train_python; IFS= read -r compiler_id; IFS= read -r train_cxx; } < "$gen/python/pgo-python.txt"
if [ -z "$train_python" ] || [ -z "$train_cxx" ]; then
    echo "CMake did not record its Python interpreter and C++ compiler" >&2
    exit 1
fi
if [ -f "$out/CMakeCache.txt" ]; then
    existing_cxx=$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "$out/CMakeCache.txt")
    if [ -n "$existing_cxx" ] && [ "$existing_cxx" != "$train_cxx" ]; then
        echo "Build directory uses a different C++ compiler; choose another directory" >&2
        exit 1
    fi
fi
rm -rf "$prof"; mkdir -p "$prof"
export ROCKETGPU_CPU_THREADS=1 ROCKETGPU_CPU_BLOCK=4
cmake --build "$gen" --config Release -j "${CMAKE_BUILD_PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN)}"
PYTHONPATH="$gen/python/Release:$gen/python${PYTHONPATH:+:$PYTHONPATH}" "$train_python" "$root/tools/pgo_train.py" ${PGO_TRAIN_ARGS:-}
if [ -n "${PGO_TRAIN_EXTRA:-}" ]; then PYTHONPATH="$gen/python/Release:$gen/python${PYTHONPATH:+:$PYTHONPATH}" bash -c "$PGO_TRAIN_EXTRA"; fi
use=$prof
case "$compiler_id" in
    GNU)
        if [ -z "$(find "$prof" -type f -name '*.gcda' -size +0c -print -quit)" ]; then
            echo "PGO training produced no GCC profile data" >&2; exit 1
        fi
        ;;
    *Clang)
        have_profile=
        for f in "$prof"/*.profraw; do [ -s "$f" ] && have_profile=1 && break; done
        if [ -z "$have_profile" ]; then
            echo "PGO training produced no Clang profile data" >&2; exit 1
        fi
        if [ -n "${LLVM_PROFDATA:-}" ]; then
            pd=$(command -v "$LLVM_PROFDATA") || { echo "LLVM_PROFDATA is not executable" >&2; exit 1; }
        else
            cxx_real=$("$train_python" -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$train_cxx")
            pd=$(dirname "$cxx_real")/llvm-profdata
            [ -x "$pd" ] || pd=$(xcrun -f llvm-profdata 2>/dev/null || command -v llvm-profdata)
        fi
        "$pd" merge -o "$prof/merged.profdata" "$prof"/*.profraw
        use=$prof/merged.profdata
        ;;
    *) echo "Unsupported PGO compiler: $compiler_id" >&2; exit 1 ;;
esac
cmake -S "$root" -B "$out" "$@" -DCMAKE_BUILD_TYPE=Release -DROCKETGPU_BUILD_PYTHON=ON \
    -DROCKETGPU_PGO=USE -DROCKETGPU_PGO_PROFILE="$use" -DPython_EXECUTABLE="$train_python" -DCMAKE_CXX_COMPILER="$train_cxx"
cmake --build "$out" --config Release --clean-first -j "${CMAKE_BUILD_PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN)}"
