import numpy as np
import torch
import scipy.spatial.distance as ssd
import sklearn.metrics.pairwise as smp
import arguments as ag
import utils.GraphUtils as gu
import datasets as ds


def build_phenotype_graph(samples: [ds.SampleBase], normalization_kind: str = "gcn") -> [[float]]:
    vectors = []

    for sample in samples:
        vector = sample.to_vector()
        vectors.append(vector)

    sim_graph = gu.build_cosine_sim_graph(vectors, normalization_kind)
    return sim_graph


def build_phenotype_graph2(samples: [ds.SampleBase]) -> [[int]]:
    vectors = []

    for sample in samples:
        vector = sample.to_vector()
        vectors.append(vector)

    similarities = smp.cosine_similarity(vectors).astype(np.float32)
    avg = np.mean(similarities)
    similarities[similarities > avg] = 1
    similarities[similarities <= avg] = 0
    np.fill_diagonal(similarities, 0)
    return similarities


def build_masked_graphs(samples: [ds.SampleBase], class_num: int) -> [[[int]]]:
    graphs_masks = []
    for i in range(class_num):
        graph_masks = build_masked_graph(samples, i)
        graphs_masks.append(torch.from_numpy(graph_masks).to(ag.Arguments.device))

    return graphs_masks


def build_masked_graph(samples: [ds.SampleBase], label: int) -> [[int]]:
    N = len(samples)
    graph_masks = np.zeros((N, N), np.int32)
    for j in range(N):
        for k in range(N):
            if samples[j].label == label and samples[k].label == label:
                graph_masks[j][k] = 1
    return graph_masks


def build_masked_graphs2(samples: [ds.SampleBase], class_num: int, sim_graph: [[int]], test_indices: [int]) -> [[[int]]]:
    N = len(samples)

    test_labels = np.empty((N,), dtype=np.int32)
    # iterate the test set
    for i in range(N):
        if i not in test_indices:
            continue

        similarities = sim_graph[i]
        classes = np.zeros((class_num, ), dtype=np.int32)

        # iterate the training set
        for j in range(N):
            if j in test_indices:
                continue

            # statistics classes
            k = samples[j].label
            classes[k] += similarities[j]

        max_index = np.argmax(classes)
        test_labels[i] = max_index

    graphs_masks = []
    for i in range(class_num):
        graph_masks = np.zeros((N, N), np.int32)
        for j in range(N):
            for k in range(N):
                label_j = test_labels[j] if j in test_indices else samples[j].label
                label_k = test_labels[k] if k in test_indices else samples[k].label

                if label_j == i and label_k == i:
                    graph_masks[j][k] = 1

        graphs_masks.append(torch.from_numpy(graph_masks).to(ag.Arguments.device))

    return graphs_masks


def build_relationship_graph(samples: [ds.SampleBase], dtype=np.bool_) -> [[bool]]:
    N = len(samples)
    masked_graph = np.zeros((N, N), dtype)
    for i in range(N):
        for j in range(i + 1, N):
            if samples[i].label == samples[j].label:
                if dtype == np.bool_:
                    masked_graph[i][j] = masked_graph[j][i] = True
                else:
                    masked_graph[i][j] = masked_graph[j][i] = 1

    return masked_graph


def build_RA_GCN_graph1(all_samples: [ds.SampleBase], normalize_kind: str) -> [[int]]:
    """
    《RA-GCN》 4.4.1. Datasets:
    Non-imaging features are utilized for graph construction (patients with equal test scores in UPDRS and MoCA who
     have the same gender and have age difference less than 2 are connected).
    """
    N = len(all_samples)
    graph = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(i + 1, N):
            lhs = all_samples[i]
            rhs = all_samples[j]
            if isinstance(lhs, ds.PDSample) and isinstance(rhs, ds.PDSample):
                # if lhs.UPDRS == rhs.UPDRS and lhs.MoCA == rhs.MoCA and lhs.sex == rhs.sex:
                if lhs.MDS_UPDRS2 == rhs.MDS_UPDRS2 and lhs.MoCA == rhs.MoCA and lhs.sex == rhs.sex:
                    if lhs.age is None or rhs.age is None:
                        same_age = lhs.age_category == rhs.age_category
                    else:
                        same_age = abs(lhs.age - rhs.age) < 2

                    same_edu_years = abs(lhs.edu_years - rhs.edu_years) < 4
                    same_race = lhs.race == rhs.race

                    if same_age and same_edu_years and same_race:
                        graph[i][j] = graph[j][i] = 1

    # print("RA-GCN adjacent graph:")
    # relationship_graph = build_relationship_graph(all_samples)
    # gu.analyze_adjacent_graph(graph, relationship_graph)

    if normalize_kind != "":
        graph, _ = gu.normalize_adjacent_graph(graph, normalize_kind)
        graph = torch.from_numpy(graph).to(ag.Arguments.device)
    return graph


def build_RA_GCN_graph2(sample_features: [[float]], normalize_kind: str = "gcn") -> [[int]]:
    """
    《RA-GCN》
    4.1 Graph construction: formulas (5)
    4.4.2. Implementation details:
    Inspired by Parisot et al. (2017), the absolute difference between the features is used as a distance for graph
     construction, and the graphs are simple.
    """
    graph = ssd.squareform(ssd.pdist(sample_features, "cityblock")).astype(np.float32)
    # It would be worse without the following line of code.
    # adjust_adjacent_graph(graph, 25, False)
    graph[graph != 0] = 1
    # print("RA-GCN adjacent graph:")
    # gu.analyze_adjacent_graph(graph, None)

    if normalize_kind != "":
        graph, _ = gu.normalize_adjacent_graph(graph, normalize_kind)
        graph = torch.from_numpy(graph).to(ag.Arguments.device)
    return graph
