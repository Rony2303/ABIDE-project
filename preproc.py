from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets
from nilearn.image import resample_to_img
from nilearn.image import load_img
import os
import glob
import tqdm
import numpy as np
import argparse
import logging
import time

YEO17 = "YEO17"
YEO7 = "YEO7"
SHEN = "SHEN"


def fisherZ(x):
    return np.arctanh(x)


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_labels(parcellation_name):
    mask = None
    if parcellation_name == YEO17:
        yeo = datasets.fetch_atlas_yeo_2011()
        mask = yeo['thick_17']
    elif parcellation_name == YEO7:
        yeo = datasets.fetch_atlas_yeo_2011()
        mask = yeo['thick_7']
    elif parcellation_name == SHEN:
        mask = "/home/dan/HartServer/Personal Folders/Dan Amir/data/shen_parcellation/shen_2mm_268_parcellation.nii.gz"
    elif os.path.isfile(parcellation_name):
        mask = parcellation_name
    else:
        raise ValueError("Invalid parcellation name")
    return load_img(mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--net", type=str, default=YEO17)
    parser.add_argument("--ts_dir", type=str, default=None)
    parser.add_argument("--subj", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default='nilearn_cache')
    parser.add_argument("--path_format", type=str, default="*.nii.gz")
    args = parser.parse_args()

    safe_make_dir(args.out_dir)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(args.out_dir, '{}_connectome_preprocess.log'.format(time.time())),
                        filemode='w')
    logging.info("Running simple preprocess with arguments:")
    logging.info("All settings used:")
    for k, v in sorted(vars(args).items()):
        logging.info("{0}: {1}".format(k, v))

    connectome_measure = ConnectivityMeasure(kind='correlation', vectorize=True)
    labels = get_labels(args.net)
    N_preprocessed = 0
    N_avail = len(os.listdir(args.data_dir))
    for sbj_path in tqdm.tqdm(glob.glob(os.path.join(args.data_dir, args.path_format))):
        sub_dir = os.path.basename(sbj_path)
        sub_id = sub_dir.split("_")[-3]
        if args.subj is not None and (sub_id != args.subj):
            continue
        out_path = os.path.join(args.out_dir, "{}.npy".format(sub_id))
        if os.path.exists(out_path):
            logging.error("Skipping subject {}. output file already exists".format(sub_dir))
            continue

        image = load_img(sbj_path)
        image_data = image.get_fdata()
        tr = image.header['pixdim'][4]
        masker = NiftiLabelsMasker(labels_img=labels, standardize=True, memory=args.cache_dir, low_pass=None,
                                   high_pass=None, t_r=tr)
        # ts = masker.fit_transform(image, confounds=None)
        # corr_mat = connectome_measure.fit_transform([ts])
        # n_labels = corr_mat.shape[0]
        # n_lables = labels.max()

        labels_mask = load_img(labels)
        labels_mask = resample_to_img(labels_mask, image, interpolation='nearest')
        labels_mask_data = labels_mask.get_fdata()
        n_lables = labels_mask_data.max()
        # ts = np.array((n_lables, image_data.shape[-1]))
        ts = []
        mat_diagonal = []

        for i in range(1, int(n_lables) + 1):
            net_data = image_data[labels_mask_data[..., 0] == i]  # [N_voxels, N_TRs]
            mean_net = np.mean(net_data, axis=0, keepdims=True)  # [1, N_TRs]
            N_trs = mean_net.shape[1]
            net_data = (net_data - net_data.mean(axis=1, keepdims=True)) / (
                    net_data.std(axis=1, keepdims=True) * np.sqrt(N_trs) + 1e-10)
            mean_net = (mean_net - mean_net.mean()) / (mean_net.std() * np.sqrt(N_trs) + 1e-10)
            corr_vals = net_data @ mean_net.T
            diag_corr = fisherZ(corr_vals).mean()
            mat_diagonal.append(diag_corr)
            # ts[i] = mean_net[0]
            ts.append(np.squeeze(mean_net))
        corr_mat = np.array(ts) @ np.array(ts).T  # [n_labels , n_labels]
        for i in range(int(n_lables)):
            corr_mat[i, i] = mat_diagonal[i]
        triu = corr_mat[np.triu_indices(n_lables)]
        np.save(out_path, triu)
        if args.ts_dir is not None:
            safe_make_dir(args.ts_dir)
            ts_path = os.path.join(args.ts_dir, sub_dir + "_ts.npy")
            np.save(ts_path, ts)
        N_preprocessed += 1
    logging.info("Finished preprocessing {} subjects out of {} available subject dirs".format(N_preprocessed, N_avail))
