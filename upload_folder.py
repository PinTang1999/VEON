import moxing as mox
import sys

def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    mox.file.copy_parallel(src, dst)


if __name__ == '__main__':
    main()