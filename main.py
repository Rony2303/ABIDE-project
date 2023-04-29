import scipy
from sklearn.decomposition import PCA
from os.path import join as opj
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
        ax.scatter(control_data[:, 0], control_data[:, 1], s=1.5, c='mediumaquamarine', alpha=0.7,
                   label='ASD Control Data')
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
def explained_variance():
    """

    :param results_dir:
    :param hcp_features:
    :param asd_features:
    :param control_features:
    :param pca_dim:
    :return:
    """
    control_on_hcp_explained_var = np.var(hcp_pca.transform(features_control), axis=0) / np.var(features_control,
                                                                                                axis=0).sum()
    fig, ax = plt.subplots(figsize=(7, 5))

    hcp_bar = ax.bar(x=np.arange(2), height=hcp_pca.explained_variance_ratio_, width=0.35, color="steelblue")
    control_bar = ax.bar(x=np.arange(2) + 0.4, height=control_on_hcp_explained_var, width=0.35, color="sandybrown")
    ax.set_xticks(np.arange(2) + 0.2)
    ax.set_xticklabels(['PC1', 'PC2'])
    ax.set(ylabel='Explained Variance ratio')
    ax.legend((hcp_bar[0], control_bar[0]), ('HCP', 'Control'))
    ax.set_title('Explained Variance ratio of HCP-Control \non PC1, PC2 of HCP')
    plt.savefig(opj(results_dir, "explained_variance_HCP_Control.png"))
    plt.show()

    # pca = PCA(pca_dim)
    # hcp_pca = pca.fit_transform(hcp_features)
    # asd_pca = pca.transform(asd_features)
    # control_pca = pca.transform(control_features)
    # plot_name = "HCP_vs_ASD_and_CONTROL.png"
    # create_data_graph(results_dir, plot_name, hcp_pca, asd_pca, control_pca)
    # plot_name = "HCP_vs_CONTROL.png"
    # create_data_graph(results_dir, plot_name, hcp_pca, None, control_pca)
    # plot_name = "HCP_vs_ASD.png"
    # create_data_graph(results_dir, plot_name, hcp_pca, asd_pca, None)


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
     create a distribution of projections by sampling 573 points from HCP’s dataset 100 times.
     Each time calculate separately the 2 leading PCs, once from the 573 points and once from the
     rest of the HCP’s points"""
    pc1_proj = []
    pc2_proj = []
    indexes = set(range(features_hcp.shape[0]))
    for i in range(100):
        hcp_sampled_ind = set(np.random.choice(features_hcp.shape[0], 572, replace=False))
        hcp_sampled_data = features_hcp.loc[hcp_sampled_ind]
        hcp_other_data = features_hcp.loc[indexes - hcp_sampled_ind]
        hcp_sampled_data_pca = PCA(2).fit(hcp_sampled_data)
        hcp_other_data_pca = PCA(2).fit(hcp_other_data)
        curr_pc1_proj = np.sqrt(
            np.power(np.inner(hcp_sampled_data_pca.components_[0], hcp_other_data_pca.components_[0]), 2) +
            np.power(np.inner(hcp_sampled_data_pca.components_[0], hcp_other_data_pca.components_[1]), 2))
        curr_pc2_proj = np.sqrt(
            np.power(np.inner(hcp_sampled_data_pca.components_[1], hcp_other_data_pca.components_[0]), 2) +
            np.power(np.inner(hcp_sampled_data_pca.components_[1], hcp_other_data_pca.components_[1]), 2))
        pc1_proj.append(curr_pc1_proj)
        pc2_proj.append(curr_pc2_proj)
    # plot
    fig, ax = plt.subplots(figsize=(9, 9))
    kde1 = sns.kdeplot(data=pd.DataFrame(pc1_proj, columns=['Projection']), x='Projection',
                       color='steelblue', label="PC1 from HCP's Projection", ax=ax)
    mean1 = np.mean(pc1_proj)
    xs1 = kde1.lines[0].get_xdata()
    ys1 = kde1.lines[0].get_ydata()
    height1 = np.interp(mean1, xs1, ys1)
    kde1.vlines(mean1, 0, height1, color='steelblue', ls=':')
    kde1.text(mean1 + 0.001, -1, f'{np.round(mean1, 2)}', rotation=90, color='steelblue')
    kde1.fill_between(xs1, 0, ys1, facecolor='steelblue', alpha=0.1)
    kde1.text(mean1 + 0.001, -1, f'{np.round(mean1, 2)}', rotation=90, color='steelblue')

    kde2 = sns.kdeplot(data=pd.DataFrame(pc2_proj, columns=['Projection']), x='Projection',
                       color='sandybrown', label="PC2 from HCP's Projection", ax=ax)
    mean2 = np.mean(pc2_proj)
    xs2 = kde2.lines[1].get_xdata()
    ys2 = kde2.lines[1].get_ydata()
    height2 = np.interp(mean2, xs2, ys2)
    kde2.vlines(mean2, 0, height2, color='sandybrown', ls=':')
    kde2.text(mean2 - 0.0015, -1, f'{np.round(mean2, 2)}', rotation=90, color='sandybrown')
    kde2.fill_between(xs2, 0, ys2, facecolor='sandybrown', alpha=0.1)

    _ = ax.axvline(control_proj_pc1_on_hcp, color='midnightblue', label="PC1 from Control's Projection")
    _ = ax.axvline(control_proj_pc2_on_hcp, color='tomato', label="PC2 from Control's Projection")

    plt.legend()
    _ = ax.set_title("Projections of PCs from 572 points from HCP\non PCs Plane from all other HCP's points",
                     fontsize=15)
    _ = plt.xticks(rotation=45)
    plt.savefig(opj(results_dir, "control_hcp_projections.jpg"))
    plt.show()


def gen_random_vec(dists, features):
    vec = np.zeros(features.shape[1])
    for i in range(features.shape[1]):
        vec[i] = dists[i].rvs()
    return vec


def gen_dist_projections_random_vector_from_hcp():
    random_vec_proj_hcp = []
    for i in range(100):
        curr_random_vec = gen_random_vec(hcp_features_distributions, features_hcp)
        curr_random_vec /= np.linalg.norm(curr_random_vec)
        vec_proj = np.sqrt(
            np.power(np.inner(curr_random_vec, hcp_comps[0]), 2) + np.power(np.inner(curr_random_vec, hcp_comps[1]), 2))
        random_vec_proj_hcp.append(vec_proj)
    return random_vec_proj_hcp


def plot_projections_random_vector_from_hcp():
    """ check if the projections of the Control group are better than the projections of a random vector from the
     HCP’s distribution. create a distribution from the projections of 100 random vectors sampled from the
     HCP’s distribution by calculating each feature’s distribution and sampling randomly from it."""
    random_vec_proj_hcp = gen_dist_projections_random_vector_from_hcp()
    fig, ax = plt.subplots(figsize=(9, 9))
    kde = sns.kdeplot(data=pd.DataFrame(random_vec_proj_hcp, columns=['Projection']), x='Projection', color="steelblue",
                      label="Random HCP Vector Projection", ax=ax)
    mean = np.mean(random_vec_proj_hcp)
    xs = kde.lines[0].get_xdata()
    ys = kde.lines[0].get_ydata()
    height = np.interp(mean, xs, ys)
    kde.vlines(mean, 0, height, colors="steelblue")
    kde.text(mean - 0.01, -0.33, f'{np.round(mean, 2)}', color='steelblue')
    kde.fill_between(xs, 0, ys, facecolor='steelblue', alpha=0.1)
    _ = ax.axvline(control_pc1_proj, color='midnightblue', label="PC1 from Control's Projection")
    _ = ax.axvline(control_pc2_proj, color='tomato', label="PC2 from Control's Projection")
    plt.legend()
    _ = ax.set_title("Projections of a random vector from HCP's distribution\non HCP's PCs plane", fontsize=15)
    plt.savefig(opj(results_dir, "projections_random_vector_hcp.png"))
    plt.show()


def plot_projection_on_data_pcs(result_dir, plot_name, data_name, data_on_data, projected_data1, data1_name,
                                architypes_2d, projected_data2=None, data2_name=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(data_on_data[:, 0], data_on_data[:, 1], s=3.3, alpha=0.3, c='black', label=data_name)
    ax.scatter(projected_data1[:, 0], projected_data1[:, 1], s=3.3, c='blue', label=data1_name)
    title = f"{data_name} & {data1_name} Data Projected on {data_name}'s PCs"
    if data2_name is not None:
        ax.scatter(projected_data2[:, 0], projected_data2[:, 1], s=3.3, c='indianred', label=data2_name)
        title = f"{data_name} & {data1_name} & {data2_name} Data Projected on {data_name}'s PCs"
    ax.scatter(architypes_2d[:, 0], architypes_2d[:, 1], s=30, c='dimgrey', label=f"{data_name}'s ARCHITYPES")
    ax.plot([architypes_2d[0, 0], architypes_2d[1, 0]], [architypes_2d[0, 1], architypes_2d[1, 1]], linestyle='--',
            linewidth=0.2, color='black')
    ax.plot([architypes_2d[1, 0], architypes_2d[2, 0]], [architypes_2d[1, 1], architypes_2d[2, 1]], linestyle='--',
            linewidth=0.2, color='black')
    ax.plot([architypes_2d[2, 0], architypes_2d[0, 0]], [architypes_2d[2, 1], architypes_2d[0, 1]], linestyle='--',
            linewidth=0.2, color='black')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    _ = ax.legend(prop={'size': 15})
    _ = ax.set_title(title, fontsize=16)
    _ = ax.set_xlabel(f"PC 1 ({data_name})", fontsize=16)
    _ = ax.set_ylabel(f"PC 2 ({data_name})", fontsize=16)
    plt.savefig(opj(result_dir, plot_name))
    plt.show()


def pearson_corr(pc_comp_data1, pc_comp_data2, pc_num, data1_name, data2_name):
    pear_corr = np.cov(pc_comp_data1[pc_num - 1], pc_comp_data2[pc_num - 1])[0][1] / (
            np.std(pc_comp_data1[pc_num - 1]) * np.std(pc_comp_data2[pc_num - 1]))
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(pc_comp_data1[pc_num - 1], pc_comp_data2[pc_num - 1])
    _ = ax.set_title(f"Pearson Corr of {data1_name} PC{pc_num} and {data2_name} PC{pc_num} is: {pear_corr}")
    _ = ax.set_xlabel(f"PC{pc_num} {data1_name}", fontsize=16)
    _ = ax.set_ylabel(f"PC{pc_num} {data2_name}", fontsize=16)
    plt.savefig(opj(results_dir, f"pear_corr_{data1_name}_{data2_name}_PC{pc_num}.png"))
    plt.show()


def cosine_similarity():
    similarities_asd_control = np.inner(asd_comps, control_comps)
    similarities_asd_hcp = np.inner(asd_comps, hcp_comps)
    similarities_hcp_control = np.inner(control_comps, hcp_comps)
    print(f"ASD vs Control:\n {similarities_asd_control}\n"
          f"ASD vs HCP:\n {similarities_asd_hcp}\n"
          f"Control vs HCP:\n {similarities_hcp_control}\n")


def pc1_top_loadings(parti_res_path, data_path, title, graph_name):
    parti_results = np.load(parti_res_path)
    data = pd.read_csv(data_path)
    w_proj = parti_results["pca_comp_"].T
    fig, ax = plt.subplots(figsize=(8, 8), gridspec_kw={'left': .1, 'right': .9, 'bottom': .35})
    pc1 = pd.Series(w_proj[:, 0], index=data.columns)
    pc1 = pc1.loc[pc1.apply(lambda w: np.abs(w)).nlargest(20).sort_values(ascending=False).index]
    sns.barplot(x=pc1.index, y=pc1, color='tab:blue')
    plt.title(title)
    plt.xticks(rotation=90, fontsize=10)
    plt.axis('tight')
    plt.savefig((opj(results_dir, graph_name)))
    plt.show()


def age_dist():
    asd_ages = meta_df.loc[meta_df['DX_GROUP'] == 1, 'AGE_AT_SCAN'].dropna()
    control_ages = meta_df.loc[meta_df['DX_GROUP'] == 2, 'AGE_AT_SCAN'].dropna()
    fig, ax = plt.subplots(figsize=(9, 9))
    mean1 = np.mean(asd_ages)
    kde1 = sns.kdeplot(data=asd_ages, color='steelblue', label=f"ASD Ages, Mean: {np.round(mean1, 2)}", ax=ax)
    xs1 = kde1.lines[0].get_xdata()
    ys1 = kde1.lines[0].get_ydata()
    height1 = np.interp(mean1, xs1, ys1)
    kde1.vlines(mean1, 0, height1, color='steelblue', ls=':')
    kde1.fill_between(xs1, 0, ys1, facecolor='steelblue', alpha=0.1)
    # kde1.text(mean1 + 0.001, -1, f'{np.round(mean1, 2)}', rotation=90, color='steelblue')
    mean2 = np.mean(control_ages)
    kde2 = sns.kdeplot(data=control_ages, color='sandybrown', label=f"Control Ages, Mean: {np.round(mean2, 2)}", ax=ax)
    xs2 = kde2.lines[1].get_xdata()
    ys2 = kde2.lines[1].get_ydata()
    height2 = np.interp(mean2, xs2, ys2)
    kde2.vlines(mean2, 0, height2, color='sandybrown', ls=':')
    kde2.fill_between(xs2, 0, ys2, facecolor='sandybrown', alpha=0.1)
    kde2.text(mean2 - 0.0015, -1, f'{np.round(mean2, 2)}', rotation=90, color='sandybrown')
    plt.legend()
    _ = ax.set_title("Ages Distribution Of ASD & Control Groups",
                     fontsize=15)
    _ = plt.xticks(rotation=45)
    plt.savefig(opj(results_dir, "ages_dist.png"))
    plt.show()


def sex_ratio():
    fig, ax = plt.subplots(figsize=(7, 5))
    asd_bar = ax.bar(x=np.arange(2), height=(len(asd_male), len(asd_female)), width=0.3, color="steelblue")
    control_bar = ax.bar(x=np.arange(2) + 0.4, height=(len(control_male), len(control_female)), width=0.3,
                         color="sandybrown")
    ax.set_xticks(np.arange(2) + 0.2)
    ax.set_xticklabels(['Male', 'Female'])
    ax.set(ylabel='Number Of Subjects')
    ax.legend((asd_bar[0], control_bar[0]), ('ASD', 'Control'))
    ax.set_title('Male-Female ratio on ASD and Control groups')
    plt.savefig(opj(results_dir, "male_female_ratio.png"))
    plt.show()


if __name__ == '__main__':
    # %%
    path_to_features = "/sci/labs/uvhart/ronyzerkavod/data/ABIDE/preprocessed_data/"
    path_to_metadata = "C:/Users/ronyz/Yuval Hart's Lab/ABIDE_project/ABIDE-project/csv_files/Phenotypic_V1_0b_preprocessed.csv"
    results_groups_path = "/sci/labs/uvhart/ronyzerkavod/ouputs/ABIDE/asd_control_groups/groups.npz"
    results_dir = "C:/Users/ronyz/Yuval Hart's Lab/ABIDE_project/ABIDE-project/plots"
    meta_df = pd.read_csv(path_to_metadata)

    # %% only for initiation - init groups (ASD & control)
    # create_features_groups(path_to_features, path_to_metadata, results_dir)

    # %% backround data check - ages dist
    age_dist()

    # %% backround data check - sex comparison
    asd_sex = meta_df.loc[meta_df['DX_GROUP'] == 1, 'SEX'].dropna()
    control_sex = meta_df.loc[meta_df['DX_GROUP'] == 2, 'SEX'].dropna()
    asd_male = asd_sex.loc[asd_sex == 1]
    asd_female = asd_sex.loc[asd_sex == 2]
    control_male = control_sex.loc[control_sex == 1]
    control_female = control_sex.loc[control_sex == 2]
    # sex_ratio()

    # backround data check - autism level ?

    # %% reduce dimention to 2D
    features_hcp = pd.read_csv(
        "C:/Users/ronyz/Yuval Hart's Lab/ABIDE_project/ABIDE-project/csv_files/hcp_features.csv")  # /sci/labs/uvhart/ronyzerkavod/code/ABIDE_project/hcp_features.csv
    features_control = pd.read_csv(
        "C:/Users/ronyz/Yuval Hart's Lab/ABIDE_project/ABIDE-project/csv_files/control_features.csv")  # np.genfromtxt('/sci/labs/uvhart/ronyzerkavod/ouputs/ABIDE/asd_control_groups/asd_features.csv',delimiter=',')
    features_asd = pd.read_csv("C:/Users/ronyz/Yuval Hart's Lab/ABIDE_project/ABIDE-project/csv_files/asd_features.csv")
    control_pca = PCA(2).fit_transform(features_control)
    asd_pca = PCA(2).fit_transform(features_asd)
    hcp_pca = PCA(2).fit_transform(features_hcp)

    # %% plot the EV
    explained_variance()

    # %% project the control and ASD data onto the HCP data and plot the results, and project the ASD data onto the control data and plot the results
    control_architypes = np.loadtxt(
        "C:/Users/ronyz/Yuval Hart's Lab/ABIDE_project/ABIDE-project/csv_files/control_archs_orig.csv", delimiter=',')
    control_architypes_2d = asd_pca.transform(control_architypes.T)
    hcp_architypes = np.loadtxt(
        "C:/Users/ronyz/Yuval Hart's Lab/ABIDE_project/ABIDE-project/csv_files/hcp_archs_orig.csv", delimiter=',')
    hcp_architypes_2d = hcp_pca.transform(hcp_architypes.T)
    hcp_on_hcp = hcp_pca.transform(features_hcp)
    control_on_control = control_pca.transform(features_control)
    plot_projection_on_data_pcs(results_dir, plot_name="Control & ASD Projection on HCP", data_name="HCP",
                                data_on_data=hcp_on_hcp,
                                projected_data1=hcp_pca.transform(features_control), data1_name="Control",
                                architypes_2d=hcp_architypes_2d, projected_data2=hcp_pca.transform(features_asd),
                                data2_name="ASD")
    plot_projection_on_data_pcs(results_dir, plot_name="ASD Projection on Control", data_name="Control",
                                data_on_data=control_on_control,
                                projected_data1=control_pca.transform(features_asd), data1_name="ASD",
                                architypes_2d=control_architypes_2d)
    # %% cosine similarity between ASD-Control, ASD-HCP, Control-HCP
    asd_3_arch_res = np.load("./results/asd_3_arch_results.npz")
    control_3_arch_res = np.load("./results/control_results_3_arch.npz")
    hcp_res = np.load("./results/hcp_results.npz")
    asd_comps = asd_3_arch_res["pca_comp_"][:2]  # [PC1, PC2]
    control_comps = control_3_arch_res["pca_comp_"][:2]  # [PC1, PC2]
    hcp_comps = hcp_res["pca_comp_"][:2]  # [PC1, PC2]
    cosine_similarity()

    # %% calculate and plot Pearson correlation for PC1&PC2 in ASD-Control, ASD-HCP, Control-HCP
    pearson_corr(asd_comps, control_comps, 1, "ASD", "Control")
    pearson_corr(asd_comps, control_comps, 2, "ASD", "Control")
    pearson_corr(asd_comps, hcp_comps, 1, "ASD", "HCP")
    pearson_corr(asd_comps, hcp_comps, 2, "ASD", "HCP")
    pearson_corr(control_comps, hcp_comps, 1, "Control", "HCP")
    pearson_corr(control_comps, hcp_comps, 2, "Control", "HCP")

    # %% calculate projection on HCP
    control_pc1_proj = np.sqrt(np.power(np.inner(control_comps[0], hcp_comps[0]), 2) +
                               np.power(np.inner(control_comps[0], hcp_comps[1]), 2))
    control_pc2_proj = np.sqrt(np.power(np.inner(control_comps[1], hcp_comps[0]), 2) +
                               np.power(np.inner(control_comps[1], hcp_comps[1]), 2))
    asd_pc1_proj_on_control = np.sqrt(np.power(np.inner(asd_comps[0], control_comps[0]), 2) +
                                      np.power(np.inner(asd_comps[0], control_comps[1]), 2))
    asd_pc2_proj_on_control = np.sqrt(np.power(np.inner(asd_comps[1], control_comps[0]), 2) +
                                      np.power(np.inner(asd_comps[1], control_comps[1]), 2))
    print(f"Control pc1 projection on HCP {control_pc2_proj}\n"
          f"Control pc2 projection on HCP {control_pc2_proj}\n"
          f"ASD pc1 projection on Control {asd_pc1_proj_on_control}\n"
          f"ASD pc2 projection on Control {asd_pc2_proj_on_control}\n")

    # %% Projection of PCs from 572 points (as the control group size) from HCP on PCs from all other HCP's points
    control_proj_pc1_on_hcp = np.sqrt(np.power(np.inner(control_pca.components_[0], hcp_pca.components_[0]), 2) +
                                      np.power(np.inner(control_pca.components_[0], hcp_pca.components_[1]), 2))
    control_proj_pc2_on_hcp = np.sqrt(np.power(np.inner(control_pca.components_[1], hcp_pca.components_[0]), 2) +
                                      np.power(np.inner(control_pca.components_[1], hcp_pca.components_[1]), 2))
    hcp_pc_distributions()

    # %%  Get PC's Top Loadings of HCP, ASD and Control results
    hcp_parti_results = "C:/Users/ronyz/Yuval Hart's Lab/ABIDE_project/ABIDE-project/results/hcp_results.npz"
    hcp_data = "C:/Users/ronyz/Yuval Hart's Lab/ABIDE_project/ABIDE-project/csv_files/hcp_features.csv"
    hcp_title = "HCP Principle Component 1 Top Loadings"
    hcp_graph_name = "HCP_PC1_TOP_LOADINGS.png"
    pc1_top_loadings(hcp_parti_results, hcp_data, hcp_title, hcp_graph_name)
    control_parti_results = "C:/Users/ronyz/Yuval Hart's Lab/ABIDE_project/ABIDE-project/results/control_results_3_arch.npz"
    control_data = "C:/Users/ronyz/Yuval Hart's Lab/ABIDE_project/ABIDE-project/csv_files/control_features.csv"
    control_title = "Control Principle Component 1 Top Loadings"
    control_graph_name = "Control_PC1_TOP_LOADINGS.png"
    pc1_top_loadings(control_parti_results, control_data, control_title, control_graph_name)
    asd_parti_results = "C:/Users/ronyz/Yuval Hart's Lab/ABIDE_project/ABIDE-project/results/asd_3_arch_results.npz"
    asd_data = "C:/Users/ronyz/Yuval Hart's Lab/ABIDE_project/ABIDE-project/csv_files/asd_features.csv"
    asd_title = "ASD Principle Component 1 Top Loadings"
    asd_graph_name = "ASD_PC1_TOP_LOADINGS.png"
    pc1_top_loadings(asd_parti_results, asd_data, asd_title, asd_graph_name)

    # %% compare the control projection on HCP vs random vector from HCP distribution
    hcp_features_distributions = []
    for i in range(features_hcp.shape[1]):
        hist = np.histogram(features_hcp.iloc[:, i], bins=50)
        hcp_features_distributions.append(scipy.stats.rv_histogram(hist))
    plot_projections_random_vector_from_hcp()

    # %%
    abide_features = np.concatenate((np.array(features_asd), np.array(features_control)), axis=0)
    abide_male_bool = np.array(meta_df)[:, 10] == 1
    abide_male_bool = abide_male_bool[:-10]
    abide_features_male = np.array(abide_features)[abide_male_bool, :]
    abide_male_dist = []
    for i in range(abide_features_male.shape[1]):
        hist = np.histogram(pd.DataFrame(abide_features_male).iloc[:, i], bins=50)
        abide_male_dist.append(scipy.stats.rv_histogram(hist))
    male_sampeled_vecs = np.zeros((30, abide_features_male.shape[1]))
    for i in range(30):
        male_sampeled_vecs[i] = gen_random_vec(abide_male_dist, abide_features_male)
    np.savetxt("C:/Users/ronyz/Yuval Hart's Lab/ABIDE_project/ABIDE-project/csv_files/abide_rand_male_features.csv",
               male_sampeled_vecs,
               delimiter=",")
    abide_female_bool = ~abide_male_bool
    abide_features_female = np.array(abide_features)[abide_female_bool, :]
    abide_female_dist = []
    for i in range(abide_features_female.shape[1]):
        hist = np.histogram(pd.DataFrame(abide_features_female).iloc[:, i], bins=50)
        abide_female_dist.append(scipy.stats.rv_histogram(hist))
    female_sampeled_vecs = np.zeros((30, abide_features_female.shape[1]))
    for i in range(30):
        female_sampeled_vecs[i] = gen_random_vec(abide_female_dist, abide_features_female)
    np.savetxt("C:/Users/ronyz/Yuval Hart's Lab/ABIDE_project/ABIDE-project/csv_files/abide_rand_female_features.csv",
               female_sampeled_vecs,
               delimiter=",")

