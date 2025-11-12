import abc
import numpy as np
import logging as l
import typing as t
import datasets as ds
import arguments as ag


class SampleSet(object):
    def __init__(self, args: ag.Arguments):
        self.all_samples: [ds.SampleBase] = None
        self.args = args

    @abc.abstractmethod
    def _load_samples(self) -> [ds.SampleBase]:
        raise NotImplementedError

    def load_samples_and_statistics(self, logger: t.Optional[l.Logger]):
        all_samples = self._load_samples()

        if self.args.class_num == 2:
            if self.args.classes != [0, 1]:
                class_min = min(self.args.classes)
                for sample in all_samples:
                    assert isinstance(sample, ds.SampleBase), "Dismiss a warning."

                    if sample.label == class_min:
                        sample.label = 0
                    else:
                        sample.label = 1
        else:
            if self.args.classes == [0, 2, 4]:
                for sample in all_samples:
                    assert isinstance(sample, ds.SampleBase), "Dismiss a warning."

                    if sample.label == 2:
                        sample.label = 1
                    elif sample.label == 4:
                        sample.label = 2

        N = all_samples.shape[0]
        indices = np.array(list(range(N)))
        np.random.shuffle(indices)
        assert all_samples.shape[0] > 0, "Check if the path to the dataset folder correct."
        all_samples = all_samples[indices]
        all_labels = [sample.label for sample in all_samples]

        # analyse
        # 0:121 1:123 2:392
        statistics = np.bincount(all_labels)
        if logger is not None:
            logger.info(f"class numbers: {statistics}")
        else:
            print(f"class numbers: {statistics}")

        zeros = []
        min_index = np.argmin(statistics)
        while statistics[min_index] == 0:
            zeros.append(min_index)
            statistics[min_index] = np.inf
            min_index = np.argmin(statistics)

        statistics = np.array(statistics)
        if len(zeros) > 0:
            statistics[zeros] = 0
        ratios = np.zeros(self.args.class_num)
        percents = np.zeros(self.args.class_num)
        denominator = statistics[min_index]

        for i in range(self.args.class_num):
            if i in zeros or denominator == 0:
                ratios[i] = 0
                percents[i] = 0
            else:
                ratios[i] = np.round(statistics[i] / denominator, 3)
                percents[i] = np.round(statistics[i] / N * 100, 3)

        if logger is not None:
            logger.info(f"class ratios: {ratios}")
            logger.info(f"class percentages: {percents}%")
        else:
            print(f"class ratios: {ratios}")
            print(f"class percentages: {percents}%")

        self.all_samples = all_samples

    def _load_samples_in_classes(self, vSamples: [ds.SampleBase], sample_ids: [], event_index: int) -> ([ds.SampleBase], [str]):
        tSamples = []
        excluded_ids = []

        for sample_id in sample_ids:
            added = False
            for sample in vSamples:
                if sample.No == sample_id and sample.event_index == event_index:
                    if sample.label not in self.args.classes:
                        excluded_ids.append(sample_id)
                        continue

                    tSamples.append(sample)
                    added = True
            if not added:
                excluded_ids.append(sample_id)

        return tSamples, excluded_ids
