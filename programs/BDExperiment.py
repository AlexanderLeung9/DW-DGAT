import numpy as np
import programs as pg
import arguments as ag
import datasets as ds
import enums as es


class BDExperiment(pg.Experiment):
    def __init__(self, args: ag.BDArguments):
        super().__init__(args)

    def initialize(self):
        """
        Can be reentered.
        """
        if self.args.validation_mode == "":
            self.args.set_validation_mode(10)
        super().initialize()

        if self.sample_set is None:
            if self.args.business == es.EBusiness.PD:
                self.sample_set = ds.PDSampleSet(self.args)
            elif self.args.business == es.EBusiness.AD:
                self.sample_set = ds.ADSampleSet(self.args)
            else:
                raise NotImplementedError(f"business={self.args.business}")
            self.sample_set.load_samples_and_statistics(self.logger)

    @staticmethod
    def iterate_classes(experiment_func):
        if ag.Arguments.business == es.EBusiness.PD:
            ALL_CLASS_NUM = 3

            class_list = [-1, 0]
            for _ in range(ALL_CLASS_NUM):
                classes = np.array(class_list)
                classes += 1
                classes = classes % ALL_CLASS_NUM
                class_list = np.sort(classes).tolist()

                experiment_func(class_list)

        elif ag.Arguments.business == es.EBusiness.AD:
            class_list = [0, 2]
            experiment_func(class_list)

            class_list = [2, 4]
            experiment_func(class_list)

            class_list = [0, 4]
            experiment_func(class_list)

        else:
            raise NotImplementedError(f"business={ag.Arguments.business}")

    @staticmethod
    def iterate_businesses(experiment_func):
        ag.Arguments.business = es.EBusiness.PD
        ag.Arguments.classes = [0, 1, 2]
        experiment_func()

        ag.Arguments.business = es.EBusiness.AD
        ag.Arguments.classes = [0, 2, 4]
        experiment_func()
