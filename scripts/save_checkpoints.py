import argparse
import subprocess
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("nodes", nargs="+")

    args = parser.parse_args()
    dirname = "{}_checkpoints".format(args.name)
    os.makedirs(dirname, exist_ok=False)

    for node in args.nodes:
        cmd = "ssh -T {} \"cd /tmp/dps/hyper && ls\"".format(node)
        print(cmd)
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        output = p.stdout.decode()
        output = [o for o in output.split('\n') if o and not o.endswith('.zip')]
        assert len(output) == 1
        archive_name = output[0]
        print("Archive name on host {} is {}".format(node, archive_name))

        zip_stem = "{}_{}".format(archive_name, node)
        cmd = "ssh -T {} \"cd /tmp/dps/hyper && zip -r {} {}\"".format(node, zip_stem, archive_name)
        print(cmd)
        subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)

        cmd = "scp  {}:/tmp/dps/hyper/{}.zip {}".format(node, zip_stem, dirname)
        print(cmd)
        subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)

        cmd = "ssh -T {} \"cd /tmp/dps/hyper && rm -f {}.zip\"".format(node, zip_stem)
        print(cmd)
        subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)
