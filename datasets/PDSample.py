import typing as t
import arguments as ag
import datasets as ds


class PDSample(ds.SampleBase):
    def __init__(
            self, No: str, label: int, event_index: int, age: float, age_category: int, sex: int, race: t.Optional[int], edu_years: t.Optional[int],
            ESS: t.Optional[int], GDS: t.Optional[int], MoCA: t.Optional[int], REM: t.Optional[int], MDS_UPDRS1: t.Optional[int],
            MDS_UPDRS2: t.Optional[int], MDS_UPDRS3: t.Optional[int], MDS_UPDRS3_on: t.Optional[int], MDS_UPDRS4: t.Optional[int]):
        """
        MDS_UPDRS: Movement Disorder Society Unified-Parkinson Disease Rating Scale
        :param No: Patient No
        :param age: [27.8767, 86.1097]
        :param age_category: [1, 3]
        :param race: 1: White, 2: Black, 3: Asian, 4: Other (includes multi-racial)
        :param edu_years: [0, 32]
        :param ESS: Epworth Sleepiness Scale, [0, 24]
        :param GDS: Geriatric Depression Scale, [0, 15]
        :param MoCA: Montreal Cognitive Assessment, [0, 30]
        :param REM: Rapid Eye Movement, [0, 13]
        :param MDS_UPDRS1: [0, 38]
        :param MDS_UPDRS2: [0, 48]
        :param MDS_UPDRS3: [0, 100]
        :param MDS_UPDRS3_on: [0, 89]
        :param MDS_UPDRS4: [0, 18]
        """
        super().__init__(No, label)
        
        self.event_index = event_index
        self.age = age
        self.age_category = age_category
        self.sex = sex
        self.race = 0 if race is None else race
        self.edu_years = 0 if edu_years is None else edu_years
        self.ESS = 0 if ESS is None else ESS
        self.GDS = 0 if GDS is None else GDS
        self.MoCA = 0 if MoCA is None else MoCA
        self.REM = 0 if REM is None else REM
        self.MDS_UPDRS1 = 0 if MDS_UPDRS1 is None else MDS_UPDRS1
        self.MDS_UPDRS2 = 0 if MDS_UPDRS2 is None else MDS_UPDRS2
        self.MDS_UPDRS3 = 0 if MDS_UPDRS3 is None else MDS_UPDRS3
        self.MDS_UPDRS3_on = 0 if MDS_UPDRS3_on is None else MDS_UPDRS3_on
        self.MDS_UPDRS4 = 0 if MDS_UPDRS4 is None else MDS_UPDRS4

    @property
    def UPDRS(self) -> float:
        value = self.MDS_UPDRS1 + self.MDS_UPDRS2 + self.MDS_UPDRS3 + self.MDS_UPDRS3_on + self.MDS_UPDRS4
        return value

    def to_vector(self) -> [float]:
        """
        HC vs. PRO
                 age: 34	86.31±7.34	86.06±7.79	72.25±15.14	86.63±9.15	84.78±9.63	91.02±8.83	81.71±15.32	90.41±10.72
        age_category: 36	85.11±5.63	85.49±5.90	70.28±11.44	87.65±8.16	85.06±6.29	87.55±8.03	83.66±10.23	87.31±8.52

        PRO vs. PD
                 age: 21	96.12±2.81	94.53±5.05	88.98±8.32	95.48±4.93	94.88±4.55	92.73±8.07	97.52±2.07	91.53±9.71
        age_category: 22	95.50±3.75	92.87±8.05	86.64±11.90	92.87±8.69	93.74±6.59	90.97±12.05	97.81±2.62	87.92±16.96

        HC vs. PD
                 age: 6	98.95±1.37	98.57±2.35	97.14±3.84	99.45±0.68	98.63±2.22	98.07±4.07	99.28±1.10	97.86±4.57
        age_category: 6	98.95±1.37	98.57±2.35	97.14±3.84	99.17±1.46	98.63±2.22	98.07±4.07	99.28±1.10	97.86±4.57

        HC vs. PRO vs. PD
                 age: 100	84.20±4.20	74.51±5.14	0.79±0.05	89.90±3.56	72.98±7.12	78.76±6.00	74.51±5.14	87.25±2.57
        age_category: 94	85.17±3.46	75.60±5.32	0.80±0.04	89.34±3.27	74.41±6.62	80.74±5.28	75.60±5.32	87.80±2.66
        """

        # vector = [self.sex, self.age, self.edu_years, self.ESS, self.GDS, self.MoCA, self.REM,
        #           self.MDS_UPDRS1, self.MDS_UPDRS2, self.MDS_UPDRS3, self.MDS_UPDRS3_on, self.MDS_UPDRS4]
        # vector = [self.sex, self.age_category, self.edu_years, self.ESS, self.GDS, self.MoCA, self.REM,
        #           self.MDS_UPDRS1, self.MDS_UPDRS2, self.MDS_UPDRS3, self.MDS_UPDRS3_on, self.MDS_UPDRS4]
        # vector = [self.sex, self.age, self.edu_years, self.race,
        #           self.MDS_UPDRS1, self.MDS_UPDRS2, self.MDS_UPDRS3, self.MDS_UPDRS4]
        vector = [self.sex, self.age, self.edu_years, self.race, self.MoCA, self.MDS_UPDRS2]
        return vector

    @staticmethod
    def field_names() -> [str]:
        # tFieldNames = ["sex", "age", "edu_years", "race", "MDS_UPDRS1", "MDS_UPDRS2", "MDS_UPDRS3", "MDS_UPDRS4"]
        tFieldNames = ["sex", "age", "edu_years", "race", "MoCA", "MDS_UPDRS2"]
        return tFieldNames

    def __str__(self) -> str:
        description = f"{self.No}-{ag.BDArguments.EVENTS[self.event_index]}-{self.label}"
        return description
