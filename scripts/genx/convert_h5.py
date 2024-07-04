import os
import sys
sys.path.append(os.getcwd())
import h5py
import hdf5plugin
import ipdb
import scripts.genx.tools.psee_loader as psee_loader
from pathlib import Path
from utils.preprocessing import _blosc_opts

if __name__ == '__main__':
    path = '/data/zht/establish_frame/gen4/gen4_test_h5_v1'

    input_dir = Path(path)

    train_path = input_dir / 'train'
    val_path = input_dir / 'val'
    test_path = input_dir / 'test'

    for split in [train_path, val_path, test_path]:
        for npy_file in split.iterdir():
            if npy_file.suffix != '.npy':
                continue
            dat_path = npy_file.parent / (
                    npy_file.stem.split('bbox')[0] + f"td.dat")
            dat = psee_loader.PSEELoader(str(dat_path))
            eve = dat.load_n_events(dat._ev_count)
            h5_file_path = str(split/dat_path.stem.split('.dat')[0]) + '.h5'

            h5 = h5py.File(h5_file_path, 'w')
            h5.create_dataset('events', data=eve, chunks=True, **_blosc_opts(complevel=1, shuffle='byte'))
            dat_path.unlink()
            print('finish created of ' + h5_file_path)
        


