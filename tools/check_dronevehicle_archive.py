#!/usr/bin/env python3
"""Check a DroneVehicle ZIP archive for unreadable entries.

This utility reads every member with Python's ZIP64-aware ``zipfile`` module and
reports CRC/decompression failures. It makes no claim about all copies distributed
by the dataset authors; use it to validate the particular files you downloaded.

Usage
-----
    python tools/check_dronevehicle_archive.py train.zip
    python tools/check_dronevehicle_archive.py train.zip --manifest damaged.tsv
    python tools/check_dronevehicle_archive.py train.zip --extract-to /data

Exit codes
----------
    0  every entry is readable
    1  the archive has damaged entries (listed, and optionally written out)
    2  the file is not a readable zip at all
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive")
    ap.add_argument("--manifest", help="write damaged entries to this TSV")
    ap.add_argument("--extract-to", help="extract, skipping damaged entries")
    args = ap.parse_args()

    if not os.path.isfile(args.archive):
        print("no such file: %s" % args.archive)
        return 2

    size = os.path.getsize(args.archive)
    print("archive : %s" % args.archive)
    print("size    : %d bytes" % size)

    try:
        zf = zipfile.ZipFile(args.archive)
    except zipfile.BadZipFile as exc:
        print("\nnot a readable zip: %s" % exc)
        return 2

    damaged = []
    ok = 0
    with zf:
        infos = zf.infolist()
        print("entries : %d" % len(infos))
        print("checking CRCs (this reads the whole archive) ...")
        for info in infos:
            if info.is_dir():
                continue
            if args.extract_to:
                target = os.path.join(args.extract_to, info.filename)
                os.makedirs(os.path.dirname(target), exist_ok=True)
            try:
                with zf.open(info) as src:
                    if args.extract_to:
                        with open(target, "wb") as dst:
                            while True:
                                block = src.read(1 << 20)
                                if not block:
                                    break
                                dst.write(block)
                    else:
                        while src.read(1 << 20):
                            pass
                ok += 1
            except Exception as exc:  # BadZipFile / zlib.error / OSError
                kind = type(exc).__name__
                module = getattr(type(exc), "__module__", "")
                if module and module not in ("builtins", "__main__"):
                    kind = "%s.%s" % (module, kind)
                damaged.append((info.filename, info.file_size,
                                "%08x" % (info.CRC & 0xFFFFFFFF), kind))
                if args.extract_to and os.path.exists(target):
                    os.remove(target)

    print("\nreadable entries : %d" % ok)
    print("damaged entries  : %d" % len(damaged))

    if damaged:
        print("\n%-46s %12s %10s  %s" % ("path", "bytes", "crc32", "error"))
        for path, fsize, crc, err in damaged[:40]:
            print("%-46s %12d %10s  %s" % (path, fsize, crc, err))
        if len(damaged) > 40:
            print("... and %d more" % (len(damaged) - 40))

        if args.manifest:
            with open(args.manifest, "w", encoding="utf-8") as f:
                f.write("path\tfile_size\texpected_crc32\terror\n")
                for row in damaged:
                    f.write("\t".join(str(x) for x in row) + "\n")
            print("\nwrote %s" % args.manifest)

        print("\nThe 'expected_crc32' column is what the archive says the file "
              "should be.\nAny other copy of the dataset can be validated against "
              "it: a replacement file is\ncorrect if and only if its CRC32 and size "
              "both match.")
        return 1

    print("\nall entries readable - archive is intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
