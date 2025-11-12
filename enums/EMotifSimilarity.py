import enum as e


class EMotifSimilarity(e.Enum):
    # MA_GCNN
    Euclidean = 0
    # DW_DGCN
    Gaussian = 1
    # Pearson's correlation
    Correlation = 2
