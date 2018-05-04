import argparse
import json
import subprocess
import sys


def sbi_json_generator(sbi_path, num_records=sys.maxsize, mem="3g"):
    with subprocess.Popen(["print-json-from-sbi.sh", mem, "-i", sbi_path, "-n",
                           str(num_records)], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT) as print_json_from_sbi_command:
        for sbi_json_out in print_json_from_sbi_command.stdout:
            sbi_json_str = sbi_json_out.decode().strip()
            if not sbi_json_str.startswith("{"):
                continue
            yield(json.loads(sbi_json_str))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input-file", type=str, required=True, help="Input SBI to read")
    argparser.add_argument("-n", "--num-records", type=int, default=sys.maxsize,
                           help="Number of records to export from SBI")
    argparser.add_argument("--mem", type=str, default="3g", help="Max heap size")
    args = argparser.parse_args()
    sbi_jsons = sbi_json_generator(args.input_file, args.num_records, args.mem)
    for sbi_json in sbi_jsons:
        print(sbi_json)

