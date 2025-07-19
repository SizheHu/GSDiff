import os
import shutil

source_base = os.path.join("rplandata", "Data")

dirs_to_move = [
    "rplang-v3-bubble-diagram",
    "rplang-v3-withsemantics",
    "rplang-v3-withsemantics-withboundary",
    "rplang-v3-withsemantics-withboundary-v2"
]

target = "."

for dir_name in dirs_to_move:
    src_path = os.path.join(source_base, dir_name)
    dst_path = os.path.join(target, dir_name)
    shutil.move(src_path, dst_path)