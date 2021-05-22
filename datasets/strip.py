import os

RESTRICTED_EXTS = [".py", ".cpp", ".h"]

for file in os.walk(".").__next__()[2]:
    _, ext = os.path.splitext(file)
    if ext in RESTRICTED_EXTS:
        continue
    with open(file, "r") as f:
        lines = [" ".join(line.split()) for line in f]
    with open(file, "w") as f:
        f.write("\n".join(lines))
