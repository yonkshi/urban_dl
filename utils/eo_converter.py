import os
import re
import pickle
import argparse

import numpy as np
import h5py
from eolearn.core import EOTask, EOPatch

# Arg parser
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', type=str, default='data/slovenia/')
args = parser.parse_args()

DATASET_DIR = args.d

PATCHES_W = 15
PATCHES_H = 10

def main():
    # Read through all directories
    with h5py.File(DATASET_DIR + "slovenia2017.hdf5", "w") as f:

        n_patches = len(os.listdir(DATASET_DIR))
        for i, patch_name in enumerate(os.listdir(DATASET_DIR)):
            printProgressBar(i, n_patches)
            if not patch_name.startswith('eopatch'): continue  # Avoid garbage directories like .DS_Store

            patch_path = os.path.join(DATASET_DIR, patch_name)

            eo_patch = EOPatch.load(patch_path)

            h5set = f.create_group(patch_name)
            # Write raw bands data (images)
            data = eo_patch.data['BANDS']
            subset = write_to_hdf5(h5set, 'data_bands', data=data, is_timeseries=True)

            # Write mask
            data = eo_patch.mask['CLM']
            subset = write_to_hdf5(h5set, 'mask/clm',  data=data, is_timeseries=True)

            data = eo_patch.mask['IS_DATA']
            subset = write_to_hdf5(h5set, 'mask/is_data', data=data, is_timeseries=True)

            data = eo_patch.mask['VALID_DATA']
            subset = write_to_hdf5(h5set, 'mask/valid_data', data=data, is_timeseries=True)

            # Write timeless mask
            data = eo_patch.mask_timeless['LULC']
            subset = write_to_hdf5(h5set, 'mask_timeless/lulc', data=data)

            data = eo_patch.mask_timeless['VALID_COUNT']
            subset = write_to_hdf5(h5set, 'mask_timeless/valid_count', data=data)

            # Write time stamp (convert to unix timestamp UTC because h5py doesn't support time)
            datetimestamps = eo_patch.timestamp
            timestamps_unix_epoch = [ dt.timestamp() for dt in datetimestamps]
            data = np.array(timestamps_unix_epoch)
            subset = write_to_hdf5(h5set, 'timestamp', data=data)

            # Bounding box
            bbox = eo_patch.bbox
            # TODO Double check if grid is y axis major or x axis major
            data = np.array([bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y])
            write_to_hdf5(h5set, 'bbox', data=data)

            # Metadata
            metadict = dict(eo_patch.meta_info)
            data = np.void(pickle.dumps(metadict))
            write_to_hdf5(h5set, 'pickled_metadata', data=data)



def write_to_hdf5(dset,
                  subset_name,
                  data,
                is_timeseries=False,
                  ):
    dtype = data.dtype
    # Chunk data by time step if it's time series
    chunk_shape = (1,) + data.shape[1:] if is_timeseries else None
    subset = dset.create_dataset(subset_name,
                                 dtype = dtype,
                                 chunks = chunk_shape,
                                 data = data,
                             )
    return subset

def extract_name(eopatch_name):
    indices = tuple(int(s) for s in re.findall(r'\d+', eopatch_name))
    assert len(indices) == 2, 'Invalid eopatch file name'
    return indices

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

if __name__ == '__main__':
    main()
