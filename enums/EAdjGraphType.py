import enum as e


class EAdjGraphType(e.Enum):
    NoGraph = 0
    Phenotype = 1
    Euclidean = 2
    # RA_GCN1
    Unweighted = 3
    # RA_GCN2
    CityBlock = 4
    Relationship = 5
