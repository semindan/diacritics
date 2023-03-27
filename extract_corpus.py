from os import listdir
from os.path import isfile, join
import argparse

def main(args):
    path_to_file = lambda file: join(args.path, file)
    is_txt_file = lambda x: (isfile(path_to_file(x)) and "txt" in x)
    text = ""
    for file_name in filter(is_txt_file, listdir(args.path)):
        with open(path_to_file(file_name), "r") as f:
            text += f.read() + "\n"

    with open(args.out_path, "w") as f:
        f.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, help="path", default="data/czech_text_document_corpus_v20"
    )
    parser.add_argument("--out_path", type=str, help="out_path", default="data/text.txt")
    args = parser.parse_args()
    main(args)
