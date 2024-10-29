import warnings

warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os

# this is needed to limit the number of scipy threads
# and let spikeinterface handle parallelization
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from pathlib import Path
import shutil
import json
import argparse
import time
from datetime import datetime, timedelta

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as sc

from spikeinterface.core.core_tools import check_json

# AIND
from aind_data_schema.core.processing import DataProcess

URL = "https://github.com/AllenNeuralDynamics/aind-ephys-postprocessing"
VERSION = "1.0"

data_folder = Path("../data/")
scratch_folder = Path("../scratch")
results_folder = Path("../results/")

# Define argument parser
parser = argparse.ArgumentParser(description="Postprocess ecephys data")

use_motion_corrected_group = parser.add_mutually_exclusive_group()
use_motion_corrected_help = (
    "If True and motion corrected has been computed (and not applied), "
    "it applies motion interpolation to the recording prior to postprocessing."
)
use_motion_corrected_group.add_argument("static_use_motion_corrected", nargs="?", default="false", help=use_motion_corrected_help)
use_motion_corrected_group.add_argument("--use-motion-corrected", action="store_true", help=use_motion_corrected_help)

skip_extensions_group = parser.add_mutually_exclusive_group()
skip_extensions_group_help = (
    "If provided, it skips the computation of the specified extensions. The argument should be a list of strings "
    "separated by commas (no spaces)."
)
use_motion_corrected_group.add_argument("static_skip_extensions", nargs="?", default="", help=skip_extensions_group_help)
use_motion_corrected_group.add_argument("--skip-extensions", default="", help=skip_extensions_group_help)

n_jobs_group = parser.add_mutually_exclusive_group()
n_jobs_help = (
    "Number of jobs to use for parallel processing. Default is 0.8 (all available cores). "
    "It can also be a float between 0 and 1 to use a fraction of available cores"
)
n_jobs_group.add_argument("static_n_jobs", nargs="?", default="-1", help=n_jobs_help)
n_jobs_group.add_argument("--n-jobs", default="-1", help=n_jobs_help)

params_group = parser.add_mutually_exclusive_group()
params_file_help = "Optional json file with parameters"
params_group.add_argument("static_params_file", nargs="?", default=None, help=params_file_help)
params_group.add_argument("--params-file", default=None, help=params_file_help)
params_group.add_argument("--params-str", default=None, help="Optional json string with parameters")


if __name__ == "__main__":
    args = parser.parse_args()

    USE_MOTION_CORRECTED = args.use_motion_corrected or args.static_use_motion_corrected == "true"
    SKIP_EXTENSIONS = args.skip_extensions or args.static_skip_extensions
    N_JOBS = args.static_n_jobs or args.n_jobs
    N_JOBS = int(N_JOBS) if not N_JOBS.startswith("0.") else float(N_JOBS)
    PARAMS_FILE = args.static_params_file or args.params_file
    PARAMS_STR = args.params_str

    if SKIP_EXTENSIONS == "":
        SKIP_EXTENSIONS = []
    else:
        SKIP_EXTENSIONS = SKIP_EXTENSIONS.split(",")

    # Use CO_CPUS env variable if available
    N_JOBS_CO = os.getenv("CO_CPUS")
    if N_JOBS_CO is not None:
        if isinstance(N_JOBS, float):
            N_JOBS = int(N_JOBS * int(N_JOBS_CO))
        elif N_JOBS == -1:
            N_JOBS = int(N_JOBS_CO)
        elif int(N_JOBS_CO) < N_JOBS:
            N_JOBS = int(N_JOBS_CO)

    print(f"N_JOBS: {N_JOBS}")

    if PARAMS_FILE is not None:
        print(f"\nUsing custom parameter file: {PARAMS_FILE}")
        with open(PARAMS_FILE, "r") as f:
            processing_params = json.load(f)
    elif PARAMS_STR is not None:
        processing_params = json.loads(PARAMS_STR)
    else:
        with open("params.json", "r") as f:
            processing_params = json.load(f)

    data_process_prefix = "data_process_postprocessing"

    job_kwargs = processing_params["job_kwargs"]
    job_kwargs["n_jobs"] = N_JOBS
    si.set_global_job_kwargs(**job_kwargs)

    postprocessing_params = processing_params["postprocessing"]
    sparsity_params = processing_params["sparsity"]
    quality_metrics_names = processing_params["quality_metrics_names"]
    quality_metrics_params = processing_params["quality_metrics"]

    ####### POSTPROCESSING ########
    print("\nPOSTPROCESSING")
    t_postprocessing_start_all = time.perf_counter()

    # check if test
    if (data_folder / "preprocessing_pipeline_output_test").is_dir():
        print("\n*******************\n**** TEST MODE ****\n*******************\n")
        preprocessed_folder = data_folder / "preprocessing_pipeline_output_test"
        spikesorted_folder = data_folder / "spikesorting_pipeline_output_test"
    else:
        preprocessed_folder = data_folder
        spikesorted_folder = data_folder

    preprocessed_folders = [p for p in preprocessed_folder.iterdir() if p.is_dir() and "preprocessed_" in p.name]

    # load job json files
    job_config_json_files = [p for p in data_folder.iterdir() if p.suffix == ".json" and "job" in p.name]
    print(f"Found {len(job_config_json_files)} json configurations")

    if len(job_config_json_files) > 0:
        recording_names = []
        for json_file in job_config_json_files:
            with open(json_file, "r") as f:
                config = json.load(f)
            recording_name = config["recording_name"]
            assert (
                preprocessed_folder / f"preprocessed_{recording_name}"
            ).is_dir(), f"Preprocessed folder for {recording_name} not found!"
            recording_names.append(recording_name)
    else:
        recording_names = [("_").join(p.name.split("_")[1:]) for p in preprocessed_folders]

    for recording_name in recording_names:
        datetime_start_postprocessing = datetime.now()
        t_postprocessing_start = time.perf_counter()
        postprocessing_notes = ""
        binary_json_file = preprocessed_folder / f"binary_{recording_name}.json"
        preprocessed_json_file = preprocessed_folder / f"preprocessed_{recording_name}.json"
        motion_corrected_folder = preprocessed_folder / f"motion_{recording_name}"

        print(f"\tProcessing {recording_name}")
        postprocessing_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"
        postprocessing_output_folder = results_folder / f"postprocessed_{recording_name}.zarr"

        try:
            recording_bin = None
            if binary_json_file.is_file():
                recording_bin = si.load_extractor(binary_json_file, base_folder=preprocessed_folder)
                print(f"\tLoaded binary recording from JSON")
            else:
                recording_bin = si.load_extractor(preprocessed_folder / f"preprocessed_{recording_name}")
            recording_lazy = None
            try:
                if preprocessed_json_file.is_file():
                    print(f"\tLoading lazy recording from JSON")
                    recording_lazy = si.load_extractor(preprocessed_json_file, base_folder=data_folder)
            except:
                print("Could not load lazy preprocessed recording")
        except ValueError as e:
            print(f"Spike sorting skipped on {recording_name}. Skipping postprocessing")
            # create an empty result file (needed for pipeline)
            postprocessing_output_folder.mkdir()
            mock_array = np.array([], dtype=bool)
            np.save(postprocessing_output_folder / "placeholder.npy", mock_array)
            continue

        if USE_MOTION_CORRECTED and motion_corrected_folder is not None:
            from spikeinterface.sortingcomponents.motion import (
                InterpolateMotionRecording,
                interpolate_motion,
            )

            print("\tCorrecting for motion prior to postprocessing")
            if not isinstance(recording, InterpolateMotionRecording):
                print("\t\tApplying motion interpolation")
                motion_info = spre.load_motion_info(motion_corrected_folder)
                interpolate_motion_kwargs = motion_info["parameters"]["interpolate_motion_kwargs"]
                recording_bin_f = spre.astype(recording_bin, "float32")

                recording_bin = interpolate_motion(
                    recording_bin_f,
                    motion=motion_info["motion"],
                    **interpolate_motion_kwargs
                )
                # InterpolateMotion is not compatible with times.
                # Removing time info for postprocessing
                for rec_segment in recording_bin._recording_segments:
                    if rec_segment.time_vector is not None:
                        rec_segment.time_vector = None

                if recording_lazy is not None:
                    recording_lazy_f = spre.astype(recording_bin, "float32")
                    recording_lazy = interpolate_motion(
                        recording_lazy_f,
                        motion=motion_info["motion"],
                        **interpolate_motion_kwargs
                    )
                    # InterpolateMotion is not compatible with times.
                    # Removing time info for postprocessing
                    for rec_segment in recording_lazy._recording_segments:
                        if rec_segment.time_vector is not None:
                            rec_segment.time_vector = None
            else:
                print("\tRecording is already interpolated")

        # make sure we have spikesorted output for the block-stream
        sorted_folder = spikesorted_folder / f"spikesorted_{recording_name}"
        if not sorted_folder.is_dir():
            raise FileNotFoundError(f"Spike sorted data for {recording_name} not found!")

        try:
            sorting = si.load_extractor(sorted_folder)
        except ValueError as e:
            print(f"Spike sorting failed on {recording_name}. Skipping postprocessing")
            # create an empty result file (needed for pipeline)
            postprocessing_output_folder.mkdir()
            mock_array = np.array([], dtype=bool)
            np.save(postprocessing_output_folder / "placeholder.npy", mock_array)
            continue

        print(f"\tCreating sorting analyzer")
        sorting_analyzer_full = si.create_sorting_analyzer(
            sorting=sorting,
            recording=recording_bin,
            sparse=True,
            return_scaled=postprocessing_params["return_scaled"],
            **sparsity_params
        )
        # compute templates for de-duplication
        # now postprocess
        analyzer_dict = postprocessing_params.copy()
        analyzer_dict.pop("duplicate_threshold")
        analyzer_dict.pop("return_scaled")
        sorting_analyzer_full.compute("random_spikes", **analyzer_dict["random_spikes"])
        sorting_analyzer_full.compute("templates")
        # de-duplication
        sorting_deduplicated = sc.remove_redundant_units(
            sorting_analyzer_full, duplicate_threshold=postprocessing_params["duplicate_threshold"]
        )
        print(
            f"\tNumber of original units: {len(sorting.unit_ids)} -- Number of units after de-duplication: {len(sorting_deduplicated.unit_ids)}"
        )
        n_duplicated = int(len(sorting.unit_ids) - len(sorting_deduplicated.unit_ids))
        postprocessing_notes += f"\n- Removed {n_duplicated} duplicated units.\n"
        deduplicated_unit_ids = sorting_deduplicated.unit_ids

        sorting_analyzer_dedup = sorting_analyzer_full.select_units(sorting_deduplicated.unit_ids)

        if recording_lazy is not None:
            recording = recording_lazy
            recording_tmp = recording_bin
        else:
            recording = recording_bin
            recording_tmp = None

        sorting_analyzer = si.create_sorting_analyzer(
            sorting=sorting_deduplicated,
            recording=recording,
            sparse=True,
            return_scaled=postprocessing_params["return_scaled"],
            sparsity=sorting_analyzer_dedup.sparsity
        )

        if recording_tmp is not None:
            print(f"\tSetting temporary binary recording")
            sorting_analyzer.set_temporary_recording(recording_tmp)

        for skip_ext in SKIP_EXTENSIONS:
            print(f"\tSkipping extension: {skip_ext}")
            analyzer_dict.pop(skip_ext, None)

        # now compute all extensions
        print(f"\tComputing all postprocessing extensions")
        sorting_analyzer.compute(analyzer_dict)

        # remove pca-based quality_metrics if principal_components is not computed
        if not "principal_components" in analyzer_dict:
            pca_metrics = sqm.get_quality_pca_metric_list()
            for metric in pca_metrics:
                if metric in quality_metrics_names:
                    quality_metrics_names.remove(metric)

        print("\tComputing quality metrics")
        qm = sorting_analyzer.compute(
            "quality_metrics",
            metric_names=quality_metrics_names,
            qm_params=quality_metrics_params
        )

        # save
        print("\tSaving SortingAnalyzer to zarr")
        sorting_analyzer = sorting_analyzer.save_as(
            format="zarr",
            folder=postprocessing_output_folder
        )

        t_postprocessing_end = time.perf_counter()
        elapsed_time_postprocessing = np.round(t_postprocessing_end - t_postprocessing_start, 2)

        # save params in output
        postprocessing_params["recording_name"] = recording_name
        postprocessing_outputs = dict(duplicated_units=n_duplicated)
        postprocessing_process = DataProcess(
            name="Ephys postprocessing",
            software_version=VERSION,  # either release or git commit
            start_date_time=datetime_start_postprocessing,
            end_date_time=datetime_start_postprocessing + timedelta(seconds=np.floor(elapsed_time_postprocessing)),
            input_location=str(data_folder),
            output_location=str(results_folder),
            code_url=URL,
            parameters=postprocessing_params,
            outputs=postprocessing_outputs,
            notes=postprocessing_notes,
        )
        with open(postprocessing_output_process_json, "w") as f:
            f.write(postprocessing_process.model_dump_json(indent=3))

    t_postprocessing_end_all = time.perf_counter()
    elapsed_time_postprocessing_all = np.round(t_postprocessing_end_all - t_postprocessing_start_all, 2)
    print(f"POSTPROCESSING time: {elapsed_time_postprocessing_all}s")
