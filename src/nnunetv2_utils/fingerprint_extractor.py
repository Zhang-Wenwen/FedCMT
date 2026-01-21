"""
    Only resolves the convention of PATH definition.
"""

from __future__ import annotations
import os
import multiprocessing
import numpy as np
from time import sleep
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from .path import get_paths


class CustomizedDatasetFingerprintExtractor(DatasetFingerprintExtractor):
    def __init__(
        self, 
        dataset_name_or_id: int, 
        path_conf: str,
        num_processes: int = 8, 
        verbose: bool = False
    ):
        self.verbose = verbose

        self.input_folder, self.preprocessed_folder, _ = get_paths(path_conf, dataset_name_or_id)
        self.dataset_name = os.path.basename(self.input_folder)
        self.num_processes = num_processes
        self.dataset_json = load_json(join(self.input_folder, 'dataset.json'))
        self.dataset = get_filenames_of_train_images_and_targets(self.input_folder, self.dataset_json)

        # We don't want to use all foreground voxels because that can accumulate a lot of data (out of memory). It is
        # also not critically important to get all pixels as long as there are enough. Let's use 10e7 voxels in total
        # (for the entire dataset)
        self.num_foreground_voxels_for_intensitystats = 10e7

    def run(self, overwrite_existing: bool = False) -> dict:
        preprocessed_output_folder = self.preprocessed_folder
        maybe_mkdir_p(preprocessed_output_folder)
        properties_file = join(preprocessed_output_folder, 'dataset_fingerprint.json')

        if not isfile(properties_file) or overwrite_existing:
            reader_writer_class = determine_reader_writer_from_dataset_json(self.dataset_json,
                                                                            # yikes. Rip the following line
                                                                            self.dataset[self.dataset.keys().__iter__().__next__()]['images'][0])

            # determine how many foreground voxels we need to sample per training case
            num_foreground_samples_per_case = int(self.num_foreground_voxels_for_intensitystats //
                                                  len(self.dataset))

            r = []
            with multiprocessing.get_context("spawn").Pool(self.num_processes) as p:
                for k in self.dataset.keys():
                    r.append(p.starmap_async(DatasetFingerprintExtractor.analyze_case,
                                             ((self.dataset[k]['images'], self.dataset[k]['label'], reader_writer_class,
                                               num_foreground_samples_per_case),)))
                remaining = list(range(len(self.dataset)))
                # p is pretty nifti. If we kill workers they just respawn but don't do any work.
                # So we need to store the original pool of workers.
                workers = [j for j in p._pool]
                with tqdm(desc=None, total=len(self.dataset), disable=self.verbose) as pbar:
                    while len(remaining) > 0:
                        all_alive = all([j.is_alive() for j in workers])
                        if not all_alive:
                            raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                               'OK jokes aside.\n'
                                               'One of your background processes is missing. This could be because of '
                                               'an error (look for an error message) or because it was killed '
                                               'by your OS due to running out of RAM. If you don\'t see '
                                               'an error message, out of RAM is likely the problem. In that case '
                                               'reducing the number of workers might help')
                        done = [i for i in remaining if r[i].ready()]
                        for _ in done:
                            pbar.update()
                        remaining = [i for i in remaining if i not in done]
                        sleep(0.1)

            # results = ptqdm(DatasetFingerprintExtractor.analyze_case,
            #                 (training_images_per_case, training_labels_per_case),
            #                 processes=self.num_processes, zipped=True, reader_writer_class=reader_writer_class,
            #                 num_samples=num_foreground_samples_per_case, disable=self.verbose)
            results = [i.get()[0] for i in r]

            shapes_after_crop = [r[0] for r in results]
            spacings = [r[1] for r in results]
            foreground_intensities_per_channel = [np.concatenate([r[2][i] for r in results]) for i in
                                                  range(len(results[0][2]))]
            foreground_intensities_per_channel = np.array(foreground_intensities_per_channel)
            # we drop this so that the json file is somewhat human readable
            # foreground_intensity_stats_by_case_and_modality = [r[3] for r in results]
            median_relative_size_after_cropping = np.median([r[4] for r in results], 0)

            num_channels = len(self.dataset_json['channel_names'].keys()
                                 if 'channel_names' in self.dataset_json.keys()
                                 else self.dataset_json['modality'].keys())
            intensity_statistics_per_channel = {}
            percentiles = np.array((0.5, 50.0, 99.5))
            for i in range(num_channels):
                percentile_00_5, median, percentile_99_5 = np.percentile(foreground_intensities_per_channel[i],
                                                                         percentiles)
                intensity_statistics_per_channel[i] = {
                    'mean': float(np.mean(foreground_intensities_per_channel[i])),
                    'median': float(median),
                    'std': float(np.std(foreground_intensities_per_channel[i])),
                    'min': float(np.min(foreground_intensities_per_channel[i])),
                    'max': float(np.max(foreground_intensities_per_channel[i])),
                    'percentile_99_5': float(percentile_99_5),
                    'percentile_00_5': float(percentile_00_5),
                }

            fingerprint = {
                    "spacings": spacings,
                    "shapes_after_crop": shapes_after_crop,
                    'foreground_intensity_properties_per_channel': intensity_statistics_per_channel,
                    "median_relative_size_after_cropping": median_relative_size_after_cropping
                }

            try:
                save_json(fingerprint, properties_file)
            except Exception as e:
                if isfile(properties_file):
                    os.remove(properties_file)
                raise e
        else:
            fingerprint = load_json(properties_file)
        return fingerprint