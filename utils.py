import sys


class Arguments(object):

    def __init__(self):
        self.parse()

    def parse(self) -> None:
        for key in sys.argv[1:len(sys.argv)-1]:
            if key.startswith("--"):
                arg = key.replace('-', '')
                try:
                    self.__setattr__(arg, sys.argv[sys.argv.index(key)+1])
                except:
                    print("expecting {0} value".format(key))
