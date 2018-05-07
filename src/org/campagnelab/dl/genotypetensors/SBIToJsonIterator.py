import argparse
import threading

import os
import ujson
import subprocess
import sys


class SbiToJsonGenerator:
    def __init__(self, sbi_path, num_records=sys.maxsize, mem="3g", sort=False, include_frequencies=False,
                 use_cache=False):
        self.sbi_path = sbi_path
        self.num_records = num_records
        self.mem = mem
        self.sort = sort
        self.process = None
        self.closed = False
        self.include_frequencies = include_frequencies
        self.use_cache = use_cache
        self.json_path = self.sbi_path + "_json_cache.json"
        self.cache_exists = os.path.isfile(self.json_path)
        self.json_cache = None

    def __enter__(self):
        print_json_from_sbi_command = ["sbi-to-json.sh", self.mem, "-i", self.sbi_path, "-n",
                                       str(self.num_records)]
        if self.sort:
            print_json_from_sbi_command.append("--sort")
        if self.include_frequencies:
            print_json_from_sbi_command.append("--include-frequency")
        if self.use_cache:
            if not self.cache_exists:
                try:
                    subprocess.run(print_json_from_sbi_command + ["-o"])
                    self.cache_exists = True
                    generated_cache = True
                except subprocess.CalledProcessError:
                    print("Unable to cache; running without cache instead")
                    self.use_cache = False
                    generated_cache = False
            else:
                generated_cache = True
            if generated_cache:
                self.json_cache = open(self.json_path)
        if not self.use_cache:
            self.process = subprocess.Popen(print_json_from_sbi_command, stdout=subprocess.PIPE, bufsize=4096*30)

    def __iter__(self):
        if (not self.use_cache and self.process is None) or (self.use_cache and self.json_cache is None):
            self.__enter__()
        if not self.use_cache:
            for sbi_json_out in self.process.stdout:
                if self.closed:
                    raise GeneratorExit

                sbi_json_str = sbi_json_out.decode().strip()
                if not sbi_json_str.startswith("{"):
                    continue
                yield (ujson.loads(sbi_json_str, precise_float=True))
        else:
            yield (ujson.loads(self.json_cache.readline(), precise_float=True))

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        if not self.use_cache:
            self.process.terminate()
            self.process.kill()
            print("Killing SbiToJsonGenerator process")
            self.process = None
        else:
            self.json_cache.close()
        self.closed = True

    def close(self):
        if not self.closed:
            self.__exit__()


def sbi_json_generator(sbi_path, num_records=sys.maxsize, mem="3g", sort=False):
    print_json_from_sbi_command = ["sbi-to-json.sh", mem, "-i", sbi_path, "-n", str(num_records)]
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
