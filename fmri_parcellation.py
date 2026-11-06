import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
import gc
import json
import pickle
from nilearn.image import mean_img, resample_to_img
from nilearn.plotting import plot_stat_map, plot_design_matrix, plot_roi, plot_design_matrix
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiLabelsMasker
from nilearn.glm.first_level import FirstLevelModel
from atlasTransform.atlasTransform.utils.atlas import load_shen_268
from nilearn.interfaces.fmriprep import load_confounds
from task_config import selected_config

# =======================================================================================================================
# # Testing
# bold_path = "./preprocessed/sub-s061/sub-s061/ses-1/func/sub-s061_ses-1_task-CCTHot_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
# mask_path = "./preprocessed/sub-s061/sub-s061/ses-1/func/sub-s061_ses-1_task-CCTHot_run-1_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
# events_path = "./ds004636/sub-s061/ses-1/func/sub-s061_ses-1_task-CCTHot_run-1_events.tsv"
# confounds_path = "./preprocessed/sub-s061/sub-s061/ses-1/func/sub-s061_ses-1_task-CCTHot_run-1_desc-confounds_timeseries.tsv"
# embedding_path = f'./Results/HyperNN_Grid_Search_Evaluation/Cross_Task_Convergence/Consensus_Embeddings/layers_1_dims_8_rank_full_emb_4_nonlinear_nodes_16_reg_0.0_cct.csv'
#
# # Obtain TR
# tr = nib.load(bold_path).header.get_zooms()[3]
#
# # Load the Shen 268 atlas and create a masker
# shen_atlas = load_shen_268(2)
# masker = NiftiLabelsMasker(labels_img=shen_atlas, mask_img=mask_path, verbose=5)
# bold_img = nib.load(bold_path)
#
# roi_time_series = masker.fit_transform(bold_img)
# print("Original BOLD shape:", bold_img.shape)
# print("Parcellated shape:", roi_time_series.shape)
#
# time = np.arange(roi_time_series.shape[0]) * tr
#
# embedding = pd.read_csv(embedding_path, index_col=0)
# selected_embedding = embedding[embedding['worker_id'] == 's061']
#
# # Load event information
# events = pd.read_csv(events_path, sep="\t")
# print(events["action"].unique())
#
# # Load motion confounds
# motion_cols = [
#     "trans_x", "trans_y", "trans_z",
#     "rot_x", "rot_y", "rot_z"
# ]
# confounds = pd.read_csv(confounds_path, sep="\t")[motion_cols].fillna(0)


# ======================================================================================================================
# fMRI Parcellation Function
# ======================================================================================================================
def batch_parcellate_shen268(project_dir):
    # Define paths
    project_dir = Path(project_dir)
    preprocessed_dir = project_dir / 'preprocessed'
    qc_path = project_dir / 'Data' / 'qc_fail_combined.csv'
    output_path = project_dir / 'Data' / 'shen268_parcellated.pkl'

    # Load QC table and identify excluded participant/task pairs
    qc = pd.read_csv(qc_path, dtype={'worker_id': str, 'task': str})
    excluded_pairs = set(qc.loc[qc['qc_fail'] == 1, ['worker_id', 'task']].itertuples(index=False, name=None))

    # Load the Shen 268 atlas
    atlas = load_shen_268(2)  # 2mm
    atlas_roi_ids = np.unique(atlas.get_fdata()).astype(int)
    atlas_roi_ids = atlas_roi_ids[atlas_roi_ids != 0]
    if len(atlas_roi_ids) != 268:
        raise ValueError(f'Expected 268 Shen regions, found {len(atlas_roi_ids)}')

    # Locate all preprocessed BOLD files in the project directory
    bold_files = sorted(preprocessed_dir.rglob('*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'))
    if not bold_files:
        raise FileNotFoundError(f'No matching preprocessed BOLD files in {preprocessed_dir}')

    # Start parcellation
    runs = {}
    for index, run_bold_path in enumerate(bold_files, start=1):
        prefix = run_bold_path.name.split('_space-')[0]
        entities = dict(part.split('-', 1) for part in prefix.split('_'))
        worker_id = entities['sub']
        session = entities['ses']
        task = entities['task']
        run = entities.get('run', '1')  # Default to run-1 if not specified
        key = (worker_id, task, session, run)

        # Exclude runs based on QC results
        if (worker_id, task) in excluded_pairs:
            print(f'==' * 50)
            print(f'[{index}/{len(bold_files)}] QC Excluded: {worker_id} - {task}', flush=True)
            print(f'==' * 50)
            continue
        if key in runs:
            raise ValueError(f'Duplicate participant/session/task/run: {key}')

        print(f'[{index}/{len(bold_files)}] Processing: {worker_id} - {task}', flush=True)

        # Define paths for the mask and JSON metadata
        run_mask_path = run_bold_path.with_name(run_bold_path.name.replace('desc-preproc_bold', 'desc-brain_mask'))
        if not run_mask_path.exists():
            raise FileNotFoundError(run_mask_path)

        # Load the BOLD and mask images
        run_bold = nib.load(str(run_bold_path))
        run_mask = nib.load(str(run_mask_path))
        tr = run_bold.header.get_zooms()[3]
        if len(run_bold.shape) != 4:
            raise ValueError(f'Expected 4D BOLD: {run_bold_path}')
        if run_bold.shape[:3] != run_mask.shape or not np.allclose(run_bold.affine, run_mask.affine):
            raise ValueError(f'BOLD/mask grid mismatch: {prefix}')

        # Define the mask
        run_masker = NiftiLabelsMasker(
            labels_img=atlas,
            mask_img=str(run_mask_path),
            standardize=None,
            verbose=0,
        )

        # Define the confounds
        run_confound_matrix, run_sample_mask = load_confounds(
            img_files=str(run_bold_path),
            strategy=('motion', 'wm_csf', 'high_pass', 'scrub'),
            motion='full',
            wm_csf='basic',
            fd_threshold=0.5,
            std_dvars_threshold=1.2,
            scrub=0
        )

        # Transform the BOLD data into regional time series (applying denoising and censoring based on confounds)
        run_signals = np.asarray(run_masker.fit_transform(run_bold, confounds=run_confound_matrix, sample_mask=run_sample_mask))

        # Check % of censored volumes
        n_total = run_bold.shape[3]
        n_censored = n_total - run_signals.shape[0]
        print(f"[{index}/{len(bold_files)}] {worker_id} - {task}: Total volumes: {run_bold.shape[3]}, "
              f"Censored volumes: {n_total - run_signals.shape[0]}, % Censored: {n_censored / n_total * 100:.2f}%")

        # Preserve actual region IDs: masking/resampling can remove parcels.
        extracted_roi_ids = [int(run_masker.region_ids_[j]) for j in range(run_signals.shape[1])]
        if len(set(extracted_roi_ids)) != len(extracted_roi_ids) or not set(extracted_roi_ids).issubset(atlas_roi_ids):
            raise ValueError(f"Unexpected extracted region IDs: {prefix}")
        aligned_signals = pd.DataFrame(run_signals, columns=extracted_roi_ids).reindex(columns=atlas_roi_ids).to_numpy(dtype=np.float32)

        # Keep a version with NaNs for censored volumes to preserve temporal alignment
        keep_mask = np.ones(n_total, dtype=bool)
        if run_sample_mask is not None:
            keep_mask[:] = False
            keep_mask[run_sample_mask] = True

        # Restore the original number of rows, with NaNs at censored volumes.
        full_signals = np.full((n_total, len(atlas_roi_ids)), np.nan, dtype=np.float32)
        full_signals[keep_mask, :] = aligned_signals

        # Store the parcellated time series and metadata for this run
        runs[key] = {
            "worker_id": worker_id,
            "task": task,
            "session": session,
            "run": run,
            "time_series": aligned_signals,
            "full_time_series": full_signals,
            "tr": tr,
            "bold_path": str(run_bold_path),
            "mask_path": str(run_mask_path),
            "keep_mask": keep_mask,
            "retained_volume_indices": np.flatnonzero(keep_mask),
            "confounds_path": str(run_bold_path.parent / f"{prefix}_desc-confounds_timeseries.tsv"),
            "missing_roi_ids": sorted(set(atlas_roi_ids.tolist()) - set(extracted_roi_ids)),
        }
        del run_bold, run_mask, run_masker, run_signals, aligned_signals
        gc.collect()

    if not runs:
        raise ValueError("No QC-eligible runs were parcellated; no pickle was saved")

    # Save pickle
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as handle:
        pickle.dump(runs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(runs)} runs: {output_path}")
    return output_path


if __name__ == "__main__":
    parcellation_pickle = batch_parcellate_shen268(Path(__file__).resolve().parent)

    with open(parcellation_pickle, "rb") as handle:
        runs = pickle.load(handle)