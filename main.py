from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os


# create data
def read_ids_from_metadata(path: str):
    metadata = pd.read_csv(path)
    asd_ids = metadata.loc[metadata['DX_GROUP'] == 1, 'SUB_ID']
    control_ids = metadata.loc[metadata['DX_GROUP'] == 2, 'SUB_ID']
    return asd_ids, control_ids


def join_subjects_to_groups(path: str, asd_ids: list, control_ids: list):
    asd_group, control_group = [], []
    for asd_id in asd_ids:
        try:
            curr_features = np.load(path + "00" + str(asd_id) + ".npy")
        except FileNotFoundError:
            continue
        asd_group.append(curr_features)
    for control_id in control_ids:
        try:
            curr_features = np.load(path + "00" + str(control_id) + ".npy")
        except FileNotFoundError:
            continue
        control_group.append(curr_features)
    return np.array(asd_group), np.array(control_group)


def create_features_groups(path_to_features, path_to_metadata, saved_dir):
    asd_ids, control_ids = read_ids_from_metadata(path_to_metadata)
    asd_group, control_group = join_subjects_to_groups(path_to_features, asd_ids, control_ids)
    np.savez(saved_dir, asd_group=asd_group, control_group=control_group)
    return asd_group, control_group


def create_data_graph(results_dir, hcp_data, asd_data=None, control_data=None):
    ax = plt.gca()
    ax.scatter(hcp_data[:, 0], hcp_data[:, 1], s=1.5, c='black', alpha=0.7)
    plt_name = "HCP_data"
    if asd_data is not None:
        ax.scatter(asd_data[:, 0], asd_data[:, 1], s=1.5, c='red', alpha=0.7)
        plt_name = "ASD_vs_HCP.png"
    if control_data is not None:
        ax.scatter(control_data[:, 0], control_data[:, 1], s=1.5, c='blue', alpha=0.7)
        plt_name = "control_vs_HCP.png"
        if asd_data is not None:
            plt_name = "ASD_and_control_vs_HCP.png"
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.xlim(-16, 16)
    plt.ylim(-8, 12)
    plt.savefig(opj(results_dir, plt_name))


# A
def abide_vs_hcp_plot(results_dir, hcp_features: np.array, asd_features: np.array, control_features: np.array,
                      pca_dim: int):
    """
    use to project the ABIDE control group onto the HCP two leading PCs'
    *project ASD
    *project control
    Parameters
    ----------
    hcp_features - all of the hcp's features
    abide_features - the ASD/Control features
    dim_pca - 2

    Returns
    -------

    """
    pca = PCA(pca_dim)
    hcp_pca = pca.fit_transform(hcp_features)
    asd_pca = pca.transform(asd_features)
    control_pca = pca.transform(control_features)
    create_data_graph(results_dir, hcp_pca, asd_pca, control_pca)
    create_data_graph(results_dir, hcp_pca, control_pca)
    create_data_graph(results_dir, hcp_pca, asd_pca)


# from here those are all statistic tests to justify the use of HCP Pareto Front
def archetypes_distributions(hcp_features: np, abide_arc: np):
    """
    use to justify the HCP pareto front with ABIDE.
    generates a distribution of the HCP’s archetypes using bootstrap (1000 time getting 573 samples).
    creating multiple archetypes sets and look at the distances of each archetype from its mean.
    compare the ABIDE Control group archetypes to the mean.
    Parameters
    ----------
    hcp_features
    abide_arc

    Returns
    -------

    """


# Analyzing HCP’s PCs Vs. ABIDE Control’s PCs
def control_pc_values():
    """ check if the projection of the Control’s 2 leading PCs on the subspace spanned by
      HCP’s 2 leading PCs is relatively close. """


def hcp_pc_distributions():
    """ To get an indication whether the values of the projections are good or not,
     create a distribution of projections by sampling 573 points from HCP’s dataset 1000 times.
     Each time calculate separately the 2 leading PCs, once from the 573 points and once from the
     rest of the HCP’s points"""


def projections_random_vector_from_hcp():
    """ check if the projections of the Control group are better than the projections of a random vector from the
     HCP’s distribution. create a distribution from the projections of 1000 random vectors sampled from the
     HCP’s distribution by calculating each feature’s distribution and sampling randomly from it."""


if __name__ == '__main__':
    path_to_features = "/sci/labs/uvhart/ronyzerkavod/code/PreProcessing/outputs/abide_features/"
    path_to_metadata = "/sci/labs/uvhart/ronyzerkavod/code/PreProcessing/Phenotypic_V1_0b_preprocessed.csv"
    results_dir = "/sci/labs/uvhart/ronyzerkavod/ouputs/ABIDE/asd_control_groups/groups.npz"
    hcp_features = pd.read_csv("/sci/labs/uvhart/ronyzerkavod/code/ABIDE_project/hcp_features.csv")
    asd_group_f, control_group_f = create_features_groups(path_to_features, path_to_metadata, results_dir)
    #abide_vs_hcp_plot(results_dir, hcp_features, asd_group_f, control_group_f, 2)
