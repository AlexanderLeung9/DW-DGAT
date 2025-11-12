import os
import csv
import numpy as np
import datasets as ds
import arguments as ag


class ADSampleSet(ds.SampleSet):
    def _load_samples(self) -> [ds.ADSample]:
        assert isinstance(self.args, ag.BDArguments)

        all_samples = []

        table_path = os.path.join(ag.Arguments.document_dir, "AD_Tables", "ADNI_merged_26Aug2024.csv")
        table_samples = ADSampleSet.__load_all_samples(table_path)

        for i in range(len(ag.BDArguments.event_parts)):
            data_folder = os.path.join(self.args.dataset_dir, ag.BDArguments.event_parts[i])
            sample_ids = []
            if os.path.exists(data_folder):
                for id_folder in os.listdir(data_folder):
                    if id_folder.isdigit():
                        sample_ids.append(id_folder)

            if not self.args.use_AD_bad2_data:
                data_folder2 = os.path.join(ag.Arguments.document_dir, "AD_Tables", f"AD_{ag.BDArguments.EVENTS[i]}_DTI_BAD2.txt")
                bad2_ids = set()
                if os.path.exists(data_folder2):
                    with open(data_folder2, 'r', encoding='utf-8') as file:
                        for tId in file:
                            bad2_ids.add(tId.strip())
                sample_ids = [tId for tId in sample_ids if tId not in bad2_ids]

            if len(sample_ids) == 0:
                all_samples.append(np.array([], dtype=ds.ADSample))
                continue

            samples, excluded_ids = self._load_samples_in_classes(table_samples, sample_ids, i)
            all_samples.append(samples)

        all_samples = np.concatenate(all_samples)

        # region check uniqueness
        N = all_samples.shape[0]
        for i in range(N):
            for j in range(i + 1, N):
                first = all_samples[i]
                second = all_samples[j]

                assert isinstance(first, ds.ADSample) and isinstance(second, ds.ADSample), "Dismiss a warning."
                id_event1 = first.No + "_" + ag.BDArguments.EVENTS[first.event_index]
                id_event2 = second.No + "_" + ag.BDArguments.EVENTS[second.event_index]
                assert id_event1 != id_event2
        # endregion

        return all_samples

    @staticmethod
    def __read_samples_info(csv_file_path: str, sample_ids: [str], event_id: str) -> {}:
        dictionary = {}

        with open(csv_file_path) as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                assert isinstance(row, dict), "Dismiss a warning."
                sample_id = row["PTID"].replace("_S_", "")
                row_event_id = row["VISCODE"]
                if sample_id in sample_ids and row_event_id == event_id:
                    dictionary[sample_id] = row

        return dictionary

    @staticmethod
    def __load_all_samples(csv_file_path: str) -> [ds.ADSample]:
        samples = []
        EVENT_IDs: [str] = ["bl", "m12", "m24"]
        LABELS: [str] = ["CN", "SMC", "EMCI", "LMCI", "AD"]
        SEX: [str] = ["Female", "Male"]
        RACES: [str] = ["Am Indian/Alaskan", "Asian", "Black", "Hawaiian/Other PI", "More than one", "Unknown", "White"]

        age_name = "AGE"
        sex_name = "PTGENDER"

        with open(csv_file_path) as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                assert isinstance(row, dict), "Dismiss a warning."
                if row["DX_bl"] == '':
                    continue
                patientNo = row["PTID"].replace("_S_", "")
                event_id = row["VISCODE"]
                label = LABELS.index(row["DX_bl"])

                try:
                    event_index = EVENT_IDs.index(event_id)
                except ValueError:
                    continue

                age = None if row[age_name] == "" else float(row[age_name])
                sex = SEX.index(row[sex_name])   # 1: male, 0: female

                edu_years = int(row["PTEDUCAT"])
                race = RACES.index(row["PTRACCAT"])
                mmse = None if row["MMSE"] == "" else int(row["MMSE"])
                moca = None if row["MOCA"] == "" else int(row["MOCA"])
                adas13 = None if row["ADAS13"] == "" else float(row["ADAS13"])
                cdrsb = None if row["CDRSB"] == "" else float(row["CDRSB"])
                faq = None if row["FAQ"] == "" else int(row["FAQ"])

                sample = ds.ADSample(patientNo, label, event_index, age, sex, edu_years, race, mmse, moca, adas13, cdrsb, faq)
                samples.append(sample)

        return samples
