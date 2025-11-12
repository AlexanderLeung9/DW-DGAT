import abc
import datasets as ds


class DatasetParams(object, metaclass=abc.ABCMeta):
    def __init__(self, root_folder: str, samples: [ds.SampleBase]):
        self.root_folder: str = root_folder
        self.samples: [ds.SampleBase] = samples
