#!/usr/bin/env bash
for file in "$@"; do
  echo "=== FILE: $file ==="
  cat "$file"
  echo ""
done

