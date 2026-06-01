# This file is part PyTSEB, consisting of of high level pyTSEB scripting
# Copyright 2018 Radoslaw Guzinski and contributors listed in the README.md file.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Modified 2026: Added convention-based file discovery, auto output path,
and config-relative path resolution for streamlined UAV batch processing.
'''

import glob
from os.path import (abspath, basename, dirname, exists, isabs, isfile,
                     join, splitext)
from os import mkdir
from re import match

from pyMETRIC.PyMETRIC import PyMETRIC

# Standard input file mappings: config key -> list of filename patterns to search
INPUT_FILE_CONVENTIONS = {
    'T_R1': ['TRAD.tif', 'T_R1.tif', 'LST.tif', 'thermal.tif'],
    'VI':   ['NDVI.tif', 'VI.tif'],
    'LAI':  ['LAI_NDVI.tif', 'LAI.tif'],
    'f_c':  ['FC.tif', 'fc.tif'],
}


def apply_crop_preset(config_data, verbose=True):
    """Resolve ``crop_type=NAME`` by promoting every ``NAME.X=Y`` entry to ``X=Y``.

    A config that contains both ``crop_type=alfalfa`` and
    ``alfalfa.k_ext=0.6`` ends up with ``k_ext=0.6`` in the flat namespace
    that the rest of the model reads from. Other crops' entries
    (``corn.k_ext``, ``wheat.k_ext``, …) are left untouched.

    Mutates ``config_data`` in place; also returns it for chaining.
    """
    crop_type = str(config_data.get('crop_type', '')).strip().lower()
    if not crop_type:
        return config_data

    prefix = crop_type + '.'
    promoted = []
    for key in list(config_data.keys()):
        if key.startswith(prefix):
            bare = key[len(prefix):]
            config_data[bare] = config_data[key]
            promoted.append(bare)

    if verbose:
        if promoted:
            print('Crop preset "%s" applied: %d parameter(s) (%s)'
                  % (crop_type, len(promoted), ', '.join(sorted(promoted))))
        else:
            print('WARNING: crop_type="%s" matched no `%s*` entries in config'
                  % (crop_type, prefix))

    return config_data


class METRICConfigFileInterface():

    def __init__(self):

        self.params = {}
        self.ready = False
        self.config_dir = ''

        temp_params = {'model': 'METRIC', 'use_METRIC_resistance': 1,
                       'G_form': 0, 'water_stress': False}
        temp_model = PyMETRIC(temp_params)
        self.input_vars = temp_model._get_input_structure().keys()

    def parse_input_config(self, input_file, is_image=True):
        '''Parses the information contained in a configuration file into a dictionary.

        All relative file paths in the config are resolved relative to
        the config file's directory (not the working directory).
        '''

        if not is_image:
            print("Point time-series interface is not implemented for ESVEP!")
            return None

        # Store the config file's directory for relative path resolution
        self.config_dir = abspath(dirname(input_file))

        # Read contents of the configuration file
        config_data = dict()
        try:
            with open(input_file, 'r') as fid:
                for line in fid:
                    if match(r'\s', line):  # skip empty line
                        continue
                    elif match('#', line):  # skip comment line
                        continue
                    elif '=' in line:
                        # Remove comments in case they exist
                        line = line.split('#')[0].rstrip(' \r\n')
                        field, value = line.split('=', 1)
                        config_data[field.strip()] = value.strip()
        except IOError:
            print('Error reading ' + input_file + ' file')

        # Resolve the active crop preset, if one is named.
        apply_crop_preset(config_data, verbose=True)

        return config_data

    def _resolve_path(self, path_str):
        """Resolve a file path relative to the config file's directory.

        Absolute paths are returned unchanged. Relative paths are resolved
        relative to the directory containing the config file.
        """
        path_str = path_str.strip().strip('"')
        if not path_str or path_str == '0':
            return path_str

        # Check if it's a numeric value (constant), not a path
        try:
            float(path_str)
            return path_str
        except ValueError:
            pass

        if isabs(path_str):
            return path_str

        return abspath(join(self.config_dir, path_str))

    def _discover_input_files(self, config_data):
        """Auto-discover standard input files from the Input/ subdirectory.

        Looks for an Input/ folder next to the config file and searches
        for files matching standard naming conventions. Only sets values
        for keys not already specified in the config.
        """
        input_dir = join(self.config_dir, 'Input')
        if not exists(input_dir):
            return

        print('Auto-discovering input files in %s' % input_dir)

        for config_key, patterns in INPUT_FILE_CONVENTIONS.items():
            # Skip if already specified in config
            if config_key in config_data and config_data[config_key].strip():
                continue

            # Search for matching files
            for pattern in patterns:
                candidates = glob.glob(join(input_dir, pattern))
                if not candidates:
                    # Try case-insensitive match
                    all_files = glob.glob(join(input_dir, '*'))
                    candidates = [f for f in all_files
                                  if basename(f).lower() == pattern.lower()]

                if candidates:
                    found = candidates[0]
                    config_data[config_key] = found
                    print('  %s -> %s' % (config_key, basename(found)))
                    break

    def _auto_output_path(self, config_data):
        """Generate output file path if not specified in config.

        Creates Output/ subdirectory next to Input/ and derives the
        output filename from the parent folder name.
        """
        if 'output_file' in config_data and config_data['output_file'].strip():
            return

        # Derive name from the dataset folder
        folder_name = basename(self.config_dir)
        output_dir = join(self.config_dir, 'Output')
        if not exists(output_dir):
            mkdir(output_dir)

        output_file = join(output_dir, folder_name + '_METRIC.tif')
        config_data['output_file'] = output_file
        print('Auto output path: %s' % output_file)

    def get_data(self, config_data, is_image):
        '''Parses the parameters in a configuration file directly to METRIC
        variables for running METRIC.'''

        if not is_image:
            print("Point time-series interface is not implemented for METRIC!")
            return None

        # Auto-discover input files and output path
        self._discover_input_files(config_data)
        self._auto_output_path(config_data)

        try:
            for var_name in self.input_vars:
                try:
                    raw_val = str(config_data[var_name]).strip('"')
                    self.params[var_name] = self._resolve_path(raw_val)
                except KeyError:
                    pass

            self.params['model'] = config_data.get('model', 'pyMETRIC').strip()
            if not self.params['model']:
                self.params['model'] = 'pyMETRIC'

            if 'calc_row' not in config_data or int(config_data['calc_row']) == 0:
                self.params['calc_row'] = [0, 0]
            else:
                self.params['calc_row'] = [
                    1,
                    float(config_data['row_az'])]

            if 'water_stress' not in config_data:
                self.params['water_stress'] = False
            else:
                self.params['water_stress'] = bool(int(config_data['water_stress']))

            if int(config_data['G_form']) == 0:
                self.params['G_form'] = [[0], float(config_data['G_constant'])]
            elif int(config_data['G_form']) == 1:
                self.params['G_form'] = [[1], float(config_data['G_ratio'])]
            elif int(config_data['G_form']) == 2:
                self.params['G_form'] = [[2,
                                         float(config_data['G_amp']),
                                         float(config_data['G_phase']),
                                         float(config_data['G_shape'])],
                                         12.0]
            elif int(config_data['G_form']) == 4:
                self.params['G_form'] = [[4],
                                         (float(config_data['G_tall']),
                                          float(config_data['G_short']))]
            elif int(config_data['G_form']) == 5:
                self.params['G_form'] = [[5], None]

            # Convert air temperature to Kelvin if provided in Celsius
            T_A1_units = config_data.get('T_A1_units', 'K').strip().upper()
            if T_A1_units == 'C' and 'T_A1' in self.params:
                try:
                    t_val = float(self.params['T_A1'])
                    self.params['T_A1'] = str(t_val + 273.15)
                    print('T_A1 converted: %.2f C -> %.2f K'
                          % (t_val, t_val + 273.15))
                except ValueError:
                    pass  # It's a file path, not a scalar — leave as-is

            # reference_type is a categorical string ('tall' or 'short').
            # _resolve_path would have turned it into a bogus filesystem path;
            # re-read it from the raw config and validate.
            ref_raw = str(config_data.get('reference_type', 'tall')).strip().lower()
            if ref_raw not in ('tall', 'short'):
                print('WARNING: reference_type="%s" not recognised; '
                      'defaulting to "tall"' % ref_raw)
                ref_raw = 'tall'
            self.params['reference_type'] = ref_raw

            # Resolve output path
            self.params['output_file'] = self._resolve_path(
                config_data['output_file'])

            # Endmember mode parameters (all optional)
            endmember_mode_params = [
                'endmember_mode', 'cold_zone', 'hot_zone',
                'cold_pixel_rc', 'hot_pixel_rc',
                'cold_pixel_xy', 'hot_pixel_xy',
                'prescribed_dT_a', 'prescribed_dT_b',
                'hot_ETrF', 'cv_window_meters', 'min_lst_range',
                'endmember_zone_field'
            ]
            for param in endmember_mode_params:
                if param in config_data:
                    self.params[param] = str(config_data[param]).strip().strip('"')

            # Resolve endmember_zones path (polygon file)
            if 'endmember_zones' in config_data:
                ez_val = str(config_data['endmember_zones']).strip().strip('"')
                if ez_val:
                    self.params['endmember_zones'] = self._resolve_path(ez_val)

            self.ready = True

        except KeyError as e:
            print('Error: missing parameter ' + str(e) + ' in the input data.')
        except ValueError as e:
            print('Error: ' + str(e))

    def run(self, is_image):

        if not is_image:
            print("Point time-series interface is not implemented for METRIC!")
            return None

        if self.ready:
            if self.params['model'] == "pyMETRIC":
                model = PyMETRIC(self.params)
            else:
                print("Unknown model: " + self.params['model'] + "!")
                return None
            model.process_local_image()

        else:
            print("pyMETRIC will not be run due to errors in the input data.")
