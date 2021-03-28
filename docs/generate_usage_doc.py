# Simple helper utility to construct usage document
import glob

document_list = sorted(glob.glob("0*_*.md"))

with open("../Readme.md", "w") as out_file:
    for doc_file in document_list:
        with open(doc_file, "r") as in_file:
            text = "".join(in_file.readlines())
            text = text.replace("[//]: # ", "![]")
            out_file.write(text)
            out_file.write("\n")
