import sys
import subprocess

hosts = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(1, 33)]
directory = "/tmp/dps/hyper/"

for host in hosts:
    sub_command = "rm -r {}".format(directory)
    command = (
        "ssh -oPasswordAuthentication=no -oStrictHostKeyChecking=no "
        "-oConnectTimeout=5 -oServerAliveInterval=2 "
        "-T {host} \"{sub_command}\"".format(host=host, sub_command=sub_command)
    )

    sys.stdout.flush()
    sys.stderr.flush()

    print("Cleaning host {}".format(host))
    try:
        subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr, shell=True)
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        print(e)
