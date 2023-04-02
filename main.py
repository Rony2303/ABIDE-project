from sklearn.decomposition import PCA
from os.path import join as opj
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import csv


# create data


def read_ids_from_metadata(path: str):
    metadata = pd.read_csv(path)
    asd_ids = metadata.loc[metadata['DX_GROUP'] == 1, 'SUB_ID']
    control_ids = metadata.loc[metadata['DX_GROUP'] == 2, 'SUB_ID']
    return asd_ids, control_ids


def join_subjects_to_groups(path: str, asd_ids: list, control_ids: list):
    asd_group, control_group, updated_asd_id, updated_control_id = [], [], [], []
    for asd_id in asd_ids:
        try:
            curr_features = np.load(path + "00" + str(asd_id) + ".npy")
        except FileNotFoundError:
            continue
        asd_group.append(curr_features)
        updated_asd_id.append(asd_id)
    for control_id in control_ids:
        try:
            curr_features = np.load(path + "00" + str(control_id) + ".npy")
        except FileNotFoundError:
            continue
        control_group.append(curr_features)
        updated_control_id.append(control_id)
    df_ASD_id = pd.DataFrame(updated_asd_id)
    df_control_id = pd.DataFrame(updated_control_id)
    df_ASD_id.to_csv('/sci/labs/uvhart/ronyzerkavod/ouputs/ABIDE/asd_control_groups/ASD_ids.csv')
    df_control_id.to_csv('/sci/labs/uvhart/ronyzerkavod/ouputs/ABIDE/asd_control_groups/Control_ids.csv')
    asd_features = np.array(asd_group)
    control_features = np.array(control_group)
    np.savetxt("/sci/labs/uvhart/ronyzerkavod/ouputs/ABIDE/asd_control_groups/asd_features.csv", asd_features,
               delimiter=",")
    np.savetxt("/sci/labs/uvhart/ronyzerkavod/ouputs/ABIDE/asd_control_groups/control_features.csv", control_features,
               delimiter=",")


def create_features_groups(path_to_features, path_to_metadata, saved_dir):
    asd_ids, control_ids = read_ids_from_metadata(path_to_metadata)
    join_subjects_to_groups(path_to_features, asd_ids, control_ids)


def create_data_graph(results_dir, plot_name, hcp_data, asd_data=None, control_data=None):
    ax = plt.gca()
    ax.scatter(hcp_data[:, 0], hcp_data[:, 1], s=1.5, c='dimgrey', alpha=0.7, label='HCP Data')
    if asd_data is not None:
        ax.scatter(asd_data[:, 0], asd_data[:, 1], s=1.5, c='indianred', alpha=0.7, label='ASD Data')
    if control_data is not None:
        ax.scatter(control_data[:, 0], control_data[:, 1], s=1.5, c='mediumaquamarine', alpha=0.7, label='ASD Control Data')
    legend = ax.legend(loc='upper right', shadow=False, fontsize='x-small')
    legend.get_frame().set_facecolor('thistle')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.savefig(opj(results_dir, plot_name))
    plt.legend()
    plt.clf()


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
    plot_name = "HCP_vs_ASD_and_CONTROL.png"
    create_data_graph(results_dir, plot_name, hcp_pca, asd_pca, control_pca)
    plot_name = "HCP_vs_CONTROL.png"
    create_data_graph(results_dir, plot_name, hcp_pca, None, control_pca)
    plot_name = "HCP_vs_ASD.png"
    create_data_graph(results_dir, plot_name, hcp_pca, asd_pca, None)

    # Explained Variance

    # asd_control_on_hcp_explained_var = np.var(hcp_pca.transform(control_features), axis=0) / np.var(control_features,
    #                                                                                 axis=0).sum()
    # ax = plt.subplots(figsize=(7, 5))
    #
    # hcp_bar = ax.bar(x=np.arange(2), height=hcp_pca.explained_variance_ratio_, width=0.4)
    # asd_control_bar = ax.bar(x=np.arange(2) + 0.4, height=asd_control_on_hcp_explained_var, width=0.4)
    # ax.set_xticks(np.arange(2) + 0.2)
    # ax.set_xticklabels(['PC1', 'PC2'])
    # ax.set(ylabel='Explained Variance ratio')
    # ax.legend((hcp_bar[0], asd_control_bar[0]), ('HCP', 'ASD CONTROL'))
    #
    # ax.set_title('Explained Variance ratio of HCP-ASD CONTROL\non PC1, PC2 of HCP')
    # plt.savefig(opj(results_dir, "explained_variance_HCP_ASD-CONTROL.png"))


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
    path_to_features = "/sci/labs/uvhart/ronyzerkavod/data/ABIDE/preprocessed_data/"
    path_to_metadata = "/sci/labs/uvhart/ronyzerkavod/code/PreProcessing/Phenotypic_V1_0b_preprocessed.csv"
    results_groups_path = "/sci/labs/uvhart/ronyzerkavod/ouputs/ABIDE/asd_control_groups/groups.npz"
    results_dir = "/sci/labs/uvhart/ronyzerkavod/ouputs/ABIDE/plots/"
    hcp_features = pd.read_csv("/sci/labs/uvhart/ronyzerkavod/code/ABIDE_project/hcp_features.csv")
    # only for initiation - init groups (ASD & control)
    # create_features_groups(path_to_features, path_to_metadata, results_dir)
    features_ASD = np.genfromtxt('/sci/labs/uvhart/ronyzerkavod/ouputs/ABIDE/asd_control_groups/asd_features.csv',
                                 delimiter=',')
    features_control = np.genfromtxt(
        '/sci/labs/uvhart/ronyzerkavod/ouputs/ABIDE/asd_control_groups/control_features.csv', delimiter=',')
    abide_vs_hcp_plot(results_dir, hcp_features, features_ASD, features_control, 2)
