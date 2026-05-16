#!/bin/bash
# Copy a small subset of the source LMDB into a writable local dir so the
# source is never touched (and the LMDB lock file can be created locally).
set -euo pipefail

SRC=/scratch/07700/sjung3/genTS/aselmdb_no-EF_consolidated/mp20bat/mp20bat_neb_000.aselmdb
DST_DIR=./input_lmdb

mkdir -p "$DST_DIR"
cp "$SRC" "$DST_DIR/"
echo "Copied: $DST_DIR/$(basename "$SRC")"
ls -lh "$DST_DIR/"
