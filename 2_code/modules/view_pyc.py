import marshal
import dis
import sys

pyc_file = r"D:\MMI-MST-FORMER\2_code\modules\__pycache__\fusion_layer.cpython-310.pyc"

with open(pyc_file, "rb") as f:
    f.read(16)  # skip header (magic number, timestamp)
    code = marshal.load(f)

print("===== DISASSEMBLED BYTECODE =====")
dis.dis(code)
