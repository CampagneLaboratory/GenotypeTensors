import argparse
import ujson
import subprocess
import sys


def sbi_json_generator(sbi_path, num_records=sys.maxsize, mem="3g", sort=False):
    print_json_from_sbi_command = ["print-json-from-sbi.sh", mem, "-i", sbi_path, "-n", str(num_records)]
    if sort:
        print_json_from_sbi_command.append("--sort")
    with subprocess.Popen(print_json_from_sbi_command, stdout=subprocess.PIPE) as print_json_from_sbi_subprocess:
        for sbi_json_out in print_json_from_sbi_subprocess.stdout:
            sbi_json_str = sbi_json_out.decode().strip()
            if not sbi_json_str.startswith("{"):
                continue
            yield(ujson.loads(sbi_json_str,precise_float=True))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input-file", type=str, required=True, help="Input SBI to read")
    argparser.add_argument("-n", "--num-records", type=int, default=sys.maxsize,
                           help="Number of records to export from SBI")
    argparser.add_argument("--mem", type=str, default="3g", help="Max heap size")
    argparser.add_argument("--sort", action="store_true", help="Sort counts in each record by decreasing count.")
    args = argparser.parse_args()
    sbi_jsons = sbi_json_generator(args.input_file, args.num_records, args.mem, args.sort)
    for sbi_json in sbi_jsons:
        print(sbi_json)

