import numpy as np
import pandas as pd


class MktRegimes:
    def __init__(self, clusters):
        self.name = str(f'clusters_{clusters}')
        self.clusters = clusters
        self.regime_data = np.array([])
        self.periods = []
