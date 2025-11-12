import numpy as np
import typing as t
import arguments as ag
import datasets as ds
import enums as es


def statistics_subjects(args: ag.Arguments, vSample_set: t.Optional[ds.SampleSet], passed_labels: list[int]):
    if args.business == es.EBusiness.PD:
        args.classes = [0, 1, 2]
        sample_set = ds.PDSampleSet(args) if vSample_set is None else vSample_set
        label_to_group = {0: "HC", 1: "PRO", 2: "PD"}
        groups = ["HC", "PRO", "PD", "Total"]
    else:
        args.classes = [0, 1, 2, 3, 4]
        sample_set = ds.ADSampleSet(args) if vSample_set is None else vSample_set
        label_to_group = {0: "CN", 1: "SMC", 2: "EMCI", 3: "LMCI", 4: "AD"}
        groups = ["CN", "SMC", "EMCI", "LMCI", "AD", "Total"]

    if vSample_set is None:
        sample_set.load_samples_and_statistics(None)

    statistics = {}
    for group in groups:
        statistics[group] = {
            "Cohort": group,
            "Sex": {"M": 0, "F": 0},
            "Age": [],
            "0m": 0,
            "12m": 0,
            "24m": 0,
            "48m": 0,
            "Total": 0
        }

    counted_subjects = set()

    for sample in sample_set.all_samples:
        if sample.label in passed_labels:
            continue

        group = label_to_group[sample.label]

        # Count Total, Sex, Age (only once).
        if sample.No not in counted_subjects:
            statistics["Total"]["Total"] += 1
            statistics["Total"]["Sex"]["M" if sample.sex == 1 else "F"] += 1
            if sample.age != 0:
                statistics["Total"]["Age"].append(sample.age)
            counted_subjects.add(sample.No)

            if group != "Unknown":
                statistics[group]["Total"] += 1
                statistics[group]["Sex"]["M" if sample.sex == 1 else "F"] += 1
                if sample.age != 0:
                    statistics[group]["Age"].append(sample.age)

        # Count each time point.
        if sample.event_index == 0:
            statistics["Total"]["0m"] += 1
            if group != "Unknown":
                statistics[group]["0m"] += 1
        elif sample.event_index == 1:
            statistics["Total"]["12m"] += 1
            if group != "Unknown":
                statistics[group]["12m"] += 1
        elif sample.event_index == 2:
            statistics["Total"]["24m"] += 1
            if group != "Unknown":
                statistics[group]["24m"] += 1
        elif sample.event_index == 3:
            statistics["Total"]["48m"] += 1
            if group != "Unknown":
                statistics[group]["48m"] += 1

    # Calculate the maximal content width to align.
    max_widths = {}
    for group in groups:
        data = statistics[group]
        age_avg = np.mean(data['Age']) if data['Age'] else 0
        age_std = np.std(data['Age']) if data['Age'] else 0
        age_str = f"{age_avg:.1f}±{age_std:.1f}"
        sex_str = f"{data['Sex']['M']}/{data['Sex']['F']}"
        max_widths[group] = max(len(data['Cohort']), len(sex_str), len(age_str), len(str(data['0m'])),
                                len(str(data['12m'])), len(str(data['24m'])), len(str(data['48m'])), len(str(data['Total'])))

    # Calculate maximum width for the label column
    label_max_width = max(len("Cohort"), len("Sex (M/F)"), len("Age (avg±std)"), len("0m"), len("12m"), len("24m"), len("48m"),
                          len("Total"))

    # output the subjects table
    build_subjects_table(label_max_width, max_widths, groups, statistics)

    # output the samples table
    build_samples_table(label_max_width, max_widths, groups, statistics)


def build_subjects_table(label_max_width, max_widths, groups, statistics):
    # --- Build the table ---
    table = list()

    # Top border
    table.append("┌" + "─" * (label_max_width + 2) + "┬" + "┬".join(["─" * (max_widths[group] + 2) for group in groups]) + "┐")

    # Header row
    header_row = "│ " + "Subjects".center(label_max_width) + " │"
    for group in groups:
        header_row += " " + group.center(max_widths[group]) + " │"
    table.append(header_row)

    # Separator
    table.append("├" + "─" * (label_max_width + 2) + "┼" + "┼".join(["─" * (max_widths[group] + 2) for group in groups]) + "┤")

    # Data rows (starting from the second label)
    labels = ["Sex (M/F)", "Age (avg±std)", "Total"]
    for label in labels:
        row = "│ " + label.ljust(label_max_width) + " │"
        for group in groups:
            data = statistics[group]
            if label == "Sex (M/F)":
                value = f"{data['Sex']['M']}/{data['Sex']['F']}"
            elif label == "Age (avg±std)":
                age_avg = np.mean(data['Age']) if data['Age'] else 0
                age_std = np.std(data['Age']) if data['Age'] else 0
                value = f"{age_avg:.1f}±{age_std:.1f}"
            elif label == "Total":
                value = len(data['Age'])
            else:
                value = data[label]
            row += " " + str(value).rjust(max_widths[group]) + " │"
        table.append(row)

    # Bottom border
    table.append("└" + "─" * (label_max_width + 2) + "┴" + "┴".join(["─" * (max_widths[group] + 2) for group in groups]) + "┘")
    print("\n".join(table))


def build_samples_table(label_max_width, max_widths, groups, statistics):
    # --- Build the table ---
    table = list()

    # Top border
    table.append("┌" + "─" * (label_max_width + 2) + "┬" + "┬".join(["─" * (max_widths[group] + 2) for group in groups]) + "┐")

    # Header row
    header_row = "│ " + "Samples".center(label_max_width) + " │"
    for group in groups:
        header_row += " " + group.center(max_widths[group]) + " │"
    table.append(header_row)

    # Separator
    table.append(
        "├" + "─" * (label_max_width + 2) + "┼" + "┼".join(["─" * (max_widths[group] + 2) for group in groups]) + "┤")

    # Data rows (starting from the second label)
    labels = ["0m", "12m", "24m", "48m", "Total"]
    for label in labels:
        row = "│ " + label.ljust(label_max_width) + " │"
        for group in groups:
            data = statistics[group]
            if label == "Total":
                value = data["0m"] + data["12m"] + data["24m"] + data["48m"]
            else:
                value = data[label]
            row += " " + str(value).rjust(max_widths[group]) + " │"
        table.append(row)

    # Bottom border
    table.append("└" + "─" * (label_max_width + 2) + "┴" + "┴".join(["─" * (max_widths[group] + 2) for group in groups]) + "┘")
    print("\n".join(table))


def main():
    ag.BDArguments.initialize_globally(0)
    args = ag.BDArguments()
    args.business = es.EBusiness.AD
    args.initialize_dataset()
    statistics_subjects(args, None, [1, 3])


if __name__ == "__main__":
    main()
