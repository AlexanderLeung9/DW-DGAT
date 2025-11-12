import typing as t
import arguments as ag
import datasets as ds


class ADSample(ds.SampleBase):
    def __init__(
            self, No: str, label: int, event_index: int, age: t.Optional[float], sex: int, edu_years: int,
            race: int, MMSE: t.Optional[int], MoCA: t.Optional[int], ADAS13: t.Optional[float], CDRSB: t.Optional[float], FAQ: t.Optional[int]):
        """
        :param No: Patient No
        :param age: Patient Age, [50.4, 91.4]
        :param edu_years: [0, 20]
        :param race: before: 0-Am Indian/Alaskan, 1-Asian, 2-Black, 3-Hawaiian/Other PI, 4-More than one, 5-Unknown, 6-White
        :param race: after: 0-Unknown, 1-Am Indian/Alaskan, 2-Asian, 3-Black, 4-Hawaiian/Other PI, 5-White, 6-More than one
        :param MMSE: Minimum Mental State Examination, [0, 30]
        :param MoCA: Montreal Cognitive Assessment, [0, 30]
        :param ADAS13: Alzheimer Disease Assessment Scale, [0, 85]
        :param CDRSB: Clinical Dementia Rating – Sum of Boxes (CDR-SB), [0, 18]
        :param FAQ: Functional Activities Questionnaire, [0, 30]
        """
        super().__init__(No, label)

        self.event_index = event_index
        self.age = 0 if age is None else age
        self.sex = sex
        self.edu_years = 0 if edu_years is None else edu_years
        if race == 0:
            self.race = 1
        elif race == 1:
            self.race = 2
        elif race == 2:
            self.race = 3
        elif race == 3:
            self.race = 4
        elif race == 4:
            self.race = 6
        elif race == 5:
            self.race = 0
        elif race == 6:
            self.race = 5
        else:
            raise NotImplementedError(f"race={race}")
        self.MMSE = 0 if MMSE is None else MMSE
        self.MoCA = 0 if MoCA is None else MoCA
        self.ADAS13 = 0 if ADAS13 is None else ADAS13
        self.CDRSB = 0 if CDRSB is None else CDRSB
        self.FAQ = 0 if FAQ is None else FAQ

    def to_vector(self) -> [float]:
        # vector = [self.sex, self.age, self.edu_years, self.race, self.MMSE, self.MoCA, self.ADAS13, self.CDRSB, self.FAQ]
        # vector = [self.sex, self.age, self.edu_years, self.race, self.MoCA, self.ADAS13]
        vector = [self.sex, self.age, self.edu_years, self.race, self.MoCA, self.MMSE]
        return vector

    @staticmethod
    def field_names() -> [str]:
        # tFieldNames = ["sex", "age", "edu_years", "race", "MoCA", "ADAS13"]
        tFieldNames = ["sex", "age", "edu_years", "race", "MoCA", "MMSE"]
        return tFieldNames

    def __str__(self) -> str:
        description = f"{self.No}-{ag.BDArguments.EVENTS[self.event_index]}-{self.label}"
        return description
