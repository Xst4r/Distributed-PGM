import os
from heapq import heappush, heappop
from distutils.core import setup

os.environ['CC'] = "g++"
src_dir = 'src'


def get_packages(src_dir):
    pkgs=[src_dir]
    h = []
    heappush(h, (0, [src_dir]))
    while h:
        try:
            prio, dir = heappop(h)
            for item in os.listdir(os.path.join(*dir)):
                if os.path.isdir(os.path.join(*dir, item)):
                    heappush(h, (prio + 1, dir + [item]))
                    pkgs.append(".".join(dir + [item]))
        except Exception as e:
            print(e)
    return pkgs

setup(
    name='Distributed PGM',
    version="0.1",
    packages=get_packages(src_dir)
)