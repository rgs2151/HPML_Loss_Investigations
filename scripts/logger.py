from typing import TextIO

class Logger:
    # redirect stdout to both file and console
    # Ref: https://stackoverflow.com/a/14906787
    def __init__(self, terminal: TextIO, file: str):
        self.terminal = terminal
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        with open(self.file, 'a') as f:
            f.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
