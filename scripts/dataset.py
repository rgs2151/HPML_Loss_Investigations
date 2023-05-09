from pathlib import Path
import numpy as np
import pandas as pd

class Dataset1:
    def __init__(self):
        # Path unless directly from somewhere else
        self.path = None

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        # Our dataset here
        return x,y
    
class Dataset2:
    def __init__(self):
        # Path unless directly from somewhere else
        self.path = None

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        # Our dataset here
        return x,y

class Dataset3:
    def __init__(self):
        # Path unless directly from somewhere else
        self.path = None

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        # Our dataset here
        return x,y
    
class Dataset4:
    def __init__(self):
        # Path unless directly from somewhere else
        self.path = None

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        # Our dataset here
        return x,y