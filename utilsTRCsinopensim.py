# utilsTRC.py — versión sin dependencias de OpenSim
# (idéntico a tu parser, pero aseguramos exponer trc_2_dict al final)

import os
import warnings
from scipy.spatial.transform import Rotation as R
import numpy as np
from numpy.lib.recfunctions import append_fields

class TRCFile(object):
    def __init__(self, fpath=None, **kwargs):
        self.marker_names = []
        if fpath is not None:
            self.read_from_file(fpath)
        else:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def read_from_file(self, fpath):
        f = open(fpath)
        first_line = f.readline().split()
        f.readline()
        third_line = f.readline().split()
        fourth_line = f.readline().split()
        f.close()

        self.path = first_line[3] if len(first_line) > 3 else ''
        self.data_rate = float(third_line[0])
        self.camera_rate = float(third_line[1])
        self.num_frames = int(third_line[2])
        self.num_markers = int(third_line[3])
        self.units = third_line[4]
        self.orig_data_rate = float(third_line[5])
        self.orig_data_start_frame = int(third_line[6])
        self.orig_num_frames = int(third_line[7])

        self.marker_names = fourth_line[2:]
        if len(self.marker_names) != self.num_markers:
            warnings.warn('Header NumMarkers no coincide con el archivo. Se ajusta al real.')
            self.num_markers = len(self.marker_names)

        col_names = ['frame_num', 'time']
        for mark in self.marker_names:
            col_names += [mark + '_tx', mark + '_ty', mark + '_tz']
        dtype = {'names': col_names, 'formats': ['int'] + ['float64'] * (3 * self.num_markers + 1)}
        usecols = [i for i in range(3 * self.num_markers + 1 + 1)]
        self.data = np.loadtxt(fpath, delimiter='\t', skiprows=5, dtype=dtype, usecols=usecols)
        self.time = self.data['time']
        n_rows = self.time.shape[0]
        if n_rows != self.num_frames:
            warnings.warn('NumFrames del header no coincide; se ajusta al real.')
            self.num_frames = n_rows

    def __getitem__(self, key):
        return self.marker(key)    
    
    def marker(self, name):
        arr = np.empty((self.num_frames, 3))
        arr[:, 0] = self.data[name + '_tx']
        arr[:, 1] = self.data[name + '_ty']
        arr[:, 2] = self.data[name + '_tz']
        return arr
    
    def add_marker(self, name, x, y, z):
        if (len(x) != self.num_frames or len(y) != self.num_frames or len(z) != self.num_frames):
            raise Exception('Dimensiones no coinciden con NumFrames.')
        self.marker_names += [name]
        self.num_markers += 1
        if not hasattr(self, 'data'):
            self.data = np.array(x, dtype=[('%s_tx' % name, 'float64')])
            self.data = append_fields(self.data, ['%s_t%s' % (name, s) for s in 'yz'], [y, z], usemask=False)
        else:
            self.data = append_fields(self.data, ['%s_t%s' % (name, s) for s in 'xyz'], [x, y, z], usemask=False)

    def marker_at(self, name, time):
        x = np.interp(time, self.time, self.data[name + '_tx'])
        y = np.interp(time, self.time, self.data[name + '_ty'])
        z = np.interp(time, self.time, self.data[name + '_tz'])
        return [x, y, z]
    
    def marker_exists(self, name):
        return name in self.marker_names
    
    def write(self, fpath):
        f = open(fpath, 'w')
        f.write('PathFileType 4\t(X/Y/Z) %s\n' % os.path.split(fpath)[0])
        f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
        f.write('%.1f\t%.1f\t%i\t%i\t%s\t%.1f\t%i\t%i\n' % (
            self.data_rate, self.camera_rate, self.num_frames, self.num_markers, self.units, self.orig_data_rate, self.orig_data_start_frame, self.orig_num_frames))
        f.write('Frame#\tTime\t')
        for imark in range(self.num_markers):
            f.write('%s\t\t\t' % self.marker_names[imark])
        f.write('\n')
        f.write('\t\t')
        for imark in range(self.num_markers):
            i = imark + 1
            f.write('X%i\tY%i\tZ%i\t' % (i, i, i))
        f.write('\n')
        f.write('\n')
        for iframe in range(self.num_frames):
            f.write('%i' % (iframe + 1))
            f.write('\t%.7f' % self.time[iframe])
            for mark in self.marker_names:
                idxs = [mark + '_tx', mark + '_ty', mark + '_tz']
                f.write('\t%.7f\t%.7f\t%.7f' % tuple(self.data[coln][iframe] for coln in idxs))
            f.write('\n')
        f.close()

    def add_noise(self, noise_width):
        for imarker in range(self.num_markers):
            components = ['_tx', '_ty', '_tz']
            for iComponent in range(3):
                noise = np.random.normal(0, noise_width, self.num_frames)
                self.data[self.marker_names[imarker] + components[iComponent]] += noise

    def rotate(self, axis, value):
        for imarker in range(self.num_markers):
            temp = np.zeros((self.num_frames, 3))
            temp[:,0] = self.data[self.marker_names[imarker] + '_tx']
            temp[:,1] = self.data[self.marker_names[imarker] + '_ty']
            temp[:,2] = self.data[self.marker_names[imarker] + '_tz']
            r = R.from_euler(axis, value, degrees=True)
            temp_rot = r.apply(temp)
            self.data[self.marker_names[imarker] + '_tx'] = temp_rot[:,0]
            self.data[self.marker_names[imarker] + '_ty'] = temp_rot[:,1]
            self.data[self.marker_names[imarker] + '_tz'] = temp_rot[:,2]

    def offset(self, axis, value):
        comp = {'x': '_tx', 'y': '_ty', 'z': '_tz'}[axis]
        for imarker in range(self.num_markers):
            self.data[self.marker_names[imarker] + comp] += value

# --- Helper que necesita utilsKinematics/gait_analysis ---
def trc_2_dict(trc_path):
    trc = TRCFile(trc_path)
    markers = {name: trc.marker(name) for name in trc.marker_names}
    return {'time': trc.time, 'markers': markers}    