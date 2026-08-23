#!/usr/bin/env python3
"""Pack a container dir into a zip with entries at the archive root (like `cd dir && zip -r out.zip ...`)."""
import os
import sys
import zipfile

src_dir, out_zip = sys.argv[1], sys.argv[2]
items = sys.argv[3:]

with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for item in items:
        path = os.path.join(src_dir, item)
        if os.path.isfile(path):
            zf.write(path, arcname=item)
        else:
            for root, dirs, files in os.walk(path):
                dirs.sort()
                for f in sorted(files):
                    full = os.path.join(root, f)
                    arc = os.path.relpath(full, src_dir)
                    zf.write(full, arcname=arc)
print("wrote", out_zip)
