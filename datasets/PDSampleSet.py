import os
import csv
import numpy as np
import datasets as ds
import arguments as ag


class PDSampleSet(ds.SampleSet):
    def _load_samples(self) -> [ds.PDSample]:
        assert isinstance(self.args, ag.BDArguments)
        all_samples = []

        new_table_path = os.path.join(ag.Arguments.document_dir, "PD_Tables", "PPMI_Curated_Data_Cut_Public_20230612_ordered.csv")
        new_all_samples = PDSampleSet.__load_all_samples(new_table_path)
        old_table_path = os.path.join(ag.Arguments.document_dir, "PD_Tables", "PPMI_Original_Cohort_BL_to_Year_5_Dataset_Apr2020.csv")
        old_all_samples = PDSampleSet.__load_all_samples(old_table_path)

        for i in range(len(ag.BDArguments.event_parts)):
            data_folder = os.path.join(self.args.dataset_dir, ag.BDArguments.event_parts[i])
            sample_ids = []
            if os.path.exists(data_folder):
                for id_folder in os.listdir(data_folder):
                    if id_folder.isdigit():
                        # bad quality
                        if "DTI" in self.args.dataset_dir:
                            if i == 0:
                                # old
                                if id_folder in ["003580", "004064"]:
                                    continue
                        sample_ids.append(id_folder)

            if len(sample_ids) == 0:
                all_samples.append(np.array([], dtype=ds.PDSample))
                continue

            samples, excluded_ids = self._load_samples_in_classes(new_all_samples, sample_ids, i)
            if len(excluded_ids) > 0:
                # PDSampleSet.__check_excluded_samples(old_table_path, excluded_ids, ag.BDArguments.EVENTS[i])
                old_samples, _ = self._load_samples_in_classes(old_all_samples, excluded_ids, i)
                samples += old_samples

            all_samples.append(samples)

        all_samples = np.concatenate(all_samples)

        # region check uniqueness
        # N = all_samples.shape[0]
        # for i in range(N):
        #     for j in range(i + 1, N):
        #         first = all_samples[i]
        #         second = all_samples[j]
        #
        #         assert isinstance(first, ds.PDSample) and isinstance(second, ds.PDSample), "Dismiss a warning."
        #         id_event1 = first.No + "_" + ag.BDArguments.EVENTS[first.event_index]
        #         id_event2 = second.No + "_" + ag.BDArguments.EVENTS[second.event_index]
        #         assert id_event1 != id_event2
        # endregion

        if self.args.balance_dataset:
            all_samples = PDSampleSet.__balance_samples(all_samples)

        return all_samples

    @staticmethod
    def __read_samples_info(csv_file_path: str, sample_ids: [str], event_id: str) -> {}:
        dictionary = {}

        with open(csv_file_path) as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                assert isinstance(row, dict), "Dismiss a warning."
                sample_id = row["PATNO"].rjust(6, '0')
                row_event_id = row["EVENT_ID"]
                if sample_id in sample_ids and row_event_id == event_id:
                    dictionary[sample_id] = row

        return dictionary

    @staticmethod
    def __load_all_samples(csv_file_path: str) -> [ds.PDSample]:
        samples = []
        EVENT_IDs: [str] = ["BL", "V04", "V06", "V10"]

        is_original = "_Original_" in csv_file_path
        if is_original:
            class_name = "APPRDX"
            age_name = "age_cat"
            sex_name = "gen"
        else:
            # For analysis purposes, the Analytic Cohort (CONCOHORT) should be used.
            # If Analytic Cohort is missing, use COHORT.
            class_name = "CONCOHORT"
            age_name = "age_at_visit"
            sex_name = "SEX"

        with open(csv_file_path) as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                assert isinstance(row, dict), "Dismiss a warning."
                patientNo = row["PATNO"].rjust(6, '0')
                event_id = row["EVENT_ID"]
                label = row[class_name]
                if label == "":
                    label = int(row["COHORT"])
                else:
                    label = int(label)

                """ original labels:
                1	PD Participant
                2	Healthy Control
                3   SWEDD (obsolete)
                4	Prodromal
                """
                if label == 1:
                    label = 2
                elif label == 2:
                    label = 0
                elif label == 4:
                    label = 1
                else:
                    label = 3
                """ adjusted labels:
                0   Healthy Control
                1   Prodromal
                2   PD Participant
                3   SWEDD
                """

                try:
                    event_index = EVENT_IDs.index(event_id)
                except ValueError:
                    continue

                if is_original:
                    age_category = int(row[age_name])
                    age = float(row["age"])
                else:
                    age = float(row[age_name])
                    if age < 56:
                        age_category = 1
                    elif 56 <= age <= 65:
                        age_category = 2
                    else:
                        age_category = 3

                sex = int(row[sex_name])   # 1: male, 0: female
                if is_original and sex == 2:
                    sex = 0

                race = None if row["race"] == "." or row["race"] == "" else int(row["race"])
                edu_years = None if row["EDUCYRS"] == "." or row["EDUCYRS"] == "" else int(row["EDUCYRS"])
                ESS = None if row["ess"] == "." or row["ess"] == "" else int(row["ess"])
                GDS = None if row["gds"] == "." or row["gds"] == "" else int(row["gds"])
                MoCA = None if row["moca"] == "." or row["moca"] == "" else int(row["moca"])
                REM = None if row["rem"] == "." or row["rem"] == "" else int(row["rem"])
                MDS_UPDRS1 = None if row["updrs1_score"] == "." or row["updrs1_score"] == "" else int(row["updrs1_score"])
                MDS_UPDRS2 = None if row["updrs2_score"] == "." or row["updrs2_score"] == "" else int(row["updrs2_score"])
                MDS_UPDRS3 = None if row["updrs3_score"] == "." or row["updrs3_score"] == "" else int(row["updrs3_score"])
                MDS_UPDRS3_on = None if row["updrs3_score_on"] == "." or row["updrs3_score_on"] == "" else int(row["updrs3_score_on"])
                MDS_UPDRS4 = None if row["updrs4_score"] == "." or row["updrs4_score"] == "" else int(row["updrs4_score"])

                sample = ds.PDSample(patientNo, label, event_index, age, age_category, sex, race, edu_years, ESS, GDS, MoCA, REM,
                                     MDS_UPDRS1, MDS_UPDRS2, MDS_UPDRS3, MDS_UPDRS3_on, MDS_UPDRS4)
                samples.append(sample)

        return samples

    @staticmethod
    def __check_excluded_samples(old_table_path: str, excluded_ids: [str], event_id: str):
        """
        APPRDX:
        1	PD Participant
        2	Healthy Control
        3	SWEDD
        4	Prodromal
        """
        samples_info = PDSampleSet.__read_samples_info(old_table_path, excluded_ids, event_id)
        pairs = [(sample_id, int(samples_info[sample_id]["APPRDX"])) for sample_id in samples_info]
        print(f"Excluded samples of {event_id} (1. PD Participant, 2. Healthy Control, 3. SWEDD, 4. Prodromal):")
        print(pairs)

    @staticmethod
    def __balance_samples(samples: [ds.SampleBase]) -> [ds.SampleBase]:
        tSamples = []
        PD_counts = [0, 0, 0]
        PD_maxes = [66, 17, 38]
        for sample in samples:
            assert isinstance(sample, ds.PDSample)
            if sample.label == 2:
                if PD_counts[sample.event_index] >= PD_maxes[sample.event_index]:
                    continue
                tSamples.append(sample)
                PD_counts[sample.event_index] += 1
            else:
                tSamples.append(sample)

        tSamples = np.array(tSamples)
        return tSamples
