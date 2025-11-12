import os
import time
import numpy as np
import utils as u
import arguments as ag
import programs as pg


class GridSearch(object):
    def __init__(self, args: ag.Arguments, grid_args: dict):
        self.experiment = pg.Experiment(args)
        self.grid_args = grid_args

        self.args_value_indices = []
        self.args_comp_num = 1
        for key in self.grid_args.keys():
            num = len(self.grid_args[key])
            self.args_value_indices.append([0, num])
            self.args_comp_num *= num

    def run(self):
        grid_args = []
        log_dir_names = []
        final_values = []
        final_value = self.experiment.final_accuracy if self.experiment.final_accuracy > 0 else self.experiment.final_loss
        last_index = len(self.args_value_indices) - 1

        for index in range(self.args_comp_num):
            print(f"The {u.cu.get_sequence_no(index+1)}/{self.args_comp_num} group:")

            i = -1
            hyperparameters = []
            for key in self.grid_args.keys():
                i += 1
                values = self.grid_args[key]
                value_index = self.args_value_indices[i][0]
                value = values[value_index]
                setattr(self.experiment.args, key, value)

                kvp = f"{key}: {value}"
                hyperparameters.append(kvp)
            str_hyperparameters = "{" + ",\t".join(hyperparameters) + "}"
            grid_args.append(str_hyperparameters)

            # region program run
            self.experiment.initialize()
            self.experiment.run()
            final_values.append(final_value)
            log_dir_name = os.path.basename(self.experiment.args.log_dir_path)
            log_dir_names.append(log_dir_name)
            # endregion

            self.__increase_index(last_index)

        # region write an index file
        index_dir_path = os.path.dirname(self.experiment.args.log_dir_path)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        index_file_path = os.path.join(index_dir_path, f"Index-{timestamp}.txt")

        final_values = np.array(final_values)
        max_indices = np.where(final_values == np.max(final_values))[0].tolist()
        min_indices = np.where(final_values == np.min(final_values))[0].tolist()
        content = []

        for i in range(self.args_comp_num):
            if i in max_indices:
                if self.experiment.final_accuracy > 0:
                    rank = " best"
                else:
                    rank = " worst"
            elif i in min_indices:
                if self.experiment.final_accuracy > 0:
                    rank = " worst"
                else:
                    rank = " best"
            else:
                rank = ""

            log_dir_name = log_dir_names[i]
            str_hyperparameters = grid_args[i]
            line = f"{log_dir_name}: {str_hyperparameters}{rank}"
            content.append(line)
        str_content = "\n".join(content)

        with open(index_file_path, "wt") as f:
            f.write(str_content)
        # endregion

    def __increase_index(self, last_index):
        self.args_value_indices[last_index][0] += 1
        if self.args_value_indices[last_index][0] == self.args_value_indices[last_index][1]:
            self.args_value_indices[last_index][0] = 0

            for j in range(last_index - 1, -1, -1):
                self.args_value_indices[j][0] += 1
                if self.args_value_indices[j][0] == self.args_value_indices[j][1]:
                    self.args_value_indices[j][0] = 0
                else:
                    break
