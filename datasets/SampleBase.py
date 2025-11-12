import typing as t


class SampleBase(object):
    def __init__(self, No: t.Union[str, int], label: int):
        """
        :param No: Sequence No or ID
        """
        self.No = No
        self.label = label

    def __str__(self) -> str:
        description = f"{self.No}-{self.label}"
        return description
