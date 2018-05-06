import argparse
import threading
import ujson
import subprocess
import sys


class SbiToJsonGenerator:
    def __init__(self, sbi_path, num_records=sys.maxsize, mem="3g", sort=False):
        self.sbi_path = sbi_path
        self.num_records = num_records
        self.mem = mem
        self.sort = sort
        self.process = None
        self.closed = False

    def __enter__(self):
        print_json_from_sbi_command = ["print-json-from-sbi.sh", self.mem, "-i", self.sbi_path, "-n",
                                       str(self.num_records)]
        if self.sort:
            print_json_from_sbi_command.append("--sort")

        self.process = subprocess.Popen(print_json_from_sbi_command, stdout=subprocess.PIPE,
                                        bufsize=1000*1024*1024) # Buffer for PIPE.

    def __iter__(self):
        if self.process is None:
            self.__enter__()

        for sbi_json_out in self.process.stdout:
            if self.closed:
                raise GeneratorExit

            sbi_json_str = sbi_json_out.decode().strip()
            if not sbi_json_str.startswith("{"):
                continue
            yield (ujson.loads(sbi_json_str, precise_float=True))

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        self.process.terminate()
        self.process.kill()
        print("Killing SbiToJsonGenerator process")
        self.process = None
        self.closed = True

    def close(self):
        if not self.closed:
            self.__exit__()


def sbi_json_generator(sbi_path, num_records=sys.maxsize, mem="3g", sort=False):
    print_json_from_sbi_command = ["print-json-from-sbi.sh", mem, "-i", sbi_path, "-n", str(num_records)]
    if sort:
        print_json_from_sbi_command.append("--sort")
    with subprocess.Popen(print_json_from_sbi_command, stdout=subprocess.PIPE) as print_json_from_sbi_subprocess:
        for sbi_json_out in print_json_from_sbi_subprocess.stdout:
            sbi_json_str = sbi_json_out.decode().strip()
            if not sbi_json_str.startswith("{"):
                continue
            yield (ujson.loads(sbi_json_str, precise_float=True))


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
