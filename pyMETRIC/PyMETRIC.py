# This file is part of pyMETRIC
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
Created on May 30 2018
@author: Radoslaw Guzinski (rmgu@dhigroup.com)

Modified 2026: Added endmember mode handling (auto/zone/manual/prescribed),
resolution-adaptive CV window, endmember quality validation, bug fixes.
'''

from collections import OrderedDict
from os.path import splitext, dirname, exists, join, basename
from os import mkdir

import numpy as np
import ast
from osgeo import gdal
from netCDF4 import Dataset

from pyTSEB.PyTSEB import PyTSEB, S_N, S_P, S_A
from pyTSEB import resistances as res
from pyTSEB import meteo_utils as met
from pyTSEB import net_radiation as rad
from pyMETRIC import METRIC, endmember_search

CIMEC = 0
ESA = 1
MINMAX = 2

# Endmember specification modes
AUTO = 'auto'
ZONE = 'zone'
MANUAL = 'manual'
PRESCRIBED = 'prescribed'

# Reference ET surface types
TALL_REFERENCE = 1   # alfalfa, Cn=66, Cd=0.25, G/Rn=0.04
SHORT_REFERENCE = 0  # grass,   Cn=37, Cd=0.24, G/Rn=0.10

VI_MAX = 0.95


class PyMETRIC(PyTSEB):

    def __init__(self, parameters):
        if "use_METRIC_resistance" not in parameters.keys():
            parameters["use_METRIC_resistance"] = 1
        if "ETrF_bare" not in parameters.keys():
            parameters["ETrF_bare"] = 0
        if "VI" not in parameters.keys():
            parameters["VI"] = ''
        if "endmember_search" not in parameters.keys():
            parameters["endmember_search"] = 0
        if "G_tall" not in parameters.keys():
            parameters["G_tall"] = 0.04
        if "G_short" not in parameters.keys():
            parameters["G_short"] = 0.10

        # Reference surface type drives ET0 flavor (alfalfa vs grass) and
        # which G ratio is used when G_form=4.
        if "reference_type" not in parameters.keys():
            parameters["reference_type"] = "tall"

        # Endmember mode defaults
        if "endmember_mode" not in parameters.keys():
            parameters["endmember_mode"] = AUTO
        if "cv_window_meters" not in parameters.keys():
            parameters["cv_window_meters"] = "5.0"
        if "min_lst_range" not in parameters.keys():
            parameters["min_lst_range"] = "3.0"

        # Optional user-provided reference ET (e.g. from a CoAgMET station).
        # etr_inst -> tall/alfalfa (used when reference_type=tall);
        # eto_inst -> short/grass  (used when reference_type=short).
        # Blank means compute reference ET internally with pet_asce.
        if "etr_inst" not in parameters.keys():
            parameters["etr_inst"] = ''
        if "eto_inst" not in parameters.keys():
            parameters["eto_inst"] = ''

        parameters["resistance_form"] = 0
        super().__init__(parameters)

        self.use_METRIC_resistance = int(self.p['use_METRIC_resistance'])
        self.endmember_search = int(self.p['endmember_search'])

    def _get_input_structure(self):
        '''Input fields' names for METRIC model.'''

        input_fields = super()._get_input_structure()
        del input_fields["leaf_width"]
        del input_fields["alpha_PT"]
        del input_fields["KN_b"]
        del input_fields["KN_c"]
        del input_fields["KN_C_dash"]

        input_fields["VI"] = "Vegetation Index"
        input_fields["ETrF_bare"] = "Reference ET fraction for bare soil (0-1)"
        input_fields["reference_type"] = "Reference surface ('tall' alfalfa or 'short' grass)"
        input_fields['alt'] = "Digital Elevation Model"
        input_fields['endmember_search'] = "Endmember search algorithm"
        input_fields['subset_output'] = "Subset coordinates for saving output"

        # New endmember mode fields
        input_fields['endmember_mode'] = "Endmember specification mode"
        input_fields['cold_zone'] = "Cold pixel search zone (xmin,ymin,xmax,ymax)"
        input_fields['hot_zone'] = "Hot pixel search zone (xmin,ymin,xmax,ymax)"
        input_fields['cold_pixel_rc'] = "Cold pixel row,col coordinates"
        input_fields['hot_pixel_rc'] = "Hot pixel row,col coordinates"
        input_fields['cold_pixel_xy'] = "Cold pixel x,y geographic coordinates"
        input_fields['hot_pixel_xy'] = "Hot pixel x,y geographic coordinates"
        input_fields['prescribed_dT_a'] = "Prescribed dT intercept"
        input_fields['prescribed_dT_b'] = "Prescribed dT slope"
        input_fields['hot_ETrF'] = "ETrF value for hot pixel"
        input_fields['cv_window_meters'] = "CV filter window size in meters"
        input_fields['min_lst_range'] = "Minimum LST range for quality warning"
        input_fields['etr_inst'] = "User-provided instantaneous ETr (mm/hr, tall/alfalfa)"
        input_fields['eto_inst'] = "User-provided instantaneous ETo (mm/hr, short/grass)"

        return input_fields

    def _set_special_model_input(self, field, dims):
        '''Special processing for setting certain input fields.'''

        # Those fields are optional for METRIC.
        if field in ["x_LAD", "f_c", "w_C"]:
            success, val = self._set_param_array(field, dims)
            if not success:
                val = np.ones(dims)
                success = True
        else:
            success = False
            val = None
        return success, val

    def _get_output_structure(self):
        '''Output fields' names for METRIC model.'''

        output_structure = OrderedDict([
            # Energy fluxes
            ('R_n1', S_P),   # net radiation reaching the surface
            ('R_ns1', S_A),  # net shortwave radiation reaching the surface
            ('R_nl1', S_A),  # net longwave radiation reaching the surface
            ('H1', S_P),  # total sensible heat flux (W/m^2)
            ('LE1', S_P),  # total latent heat flux (W/m^2)
            ('G1', S_P),  # ground heat flux (W/m^2)
            # resistances
            ('R_A1', S_A),  # Aerodynamic resistance to heat transport (s m-1)
            # miscellaneous
            ('omega0', S_N),  # nadir view vegetation clumping factor
            ('L', S_A),  # Monin Obukhov Length
            ('theta_s1', S_N),  # Sun zenith angle
            ('F', S_N),  # Leaf Area Index
            ('z_0M', S_N),  # Aerodynamic roughness length for momentum (m)
            ('d_0', S_N),  # Zero-plane displacement height (m)
            ('Skyl', S_N),
            ('L', S_A),  # Monin Obukhov Length at time t1
            ('u_friction', S_A),  # Friction velocity
            ('flag', S_A),  # Quality flag
            ('n_iterations', S_N),
            ('ETref_datum', S_A),
            ('ETref', S_A),
            ('fETr', S_A)])

        return output_structure

    def process_local_image(self):
        '''Runs pyMETRIC for all the pixels in an image.'''

        # ======================================
        # Process the input

        in_data = dict()

        if 'subset' in self.p:
            subset = ast.literal_eval(self.p['subset'])
        else:
            subset = []

        # Open the LST data
        try:
            fid = gdal.Open(self.p['T_R1'], gdal.GA_ReadOnly)
            self.prj = fid.GetProjection()
            self.geo = fid.GetGeoTransform()
            if subset:
                in_data['T_R1'] = fid.GetRasterBand(1).ReadAsArray(
                    subset[0], subset[1], subset[2], subset[3])
                self.geo = [self.geo[0] + subset[0] * self.geo[1],
                            self.geo[1], self.geo[2],
                            self.geo[3] + subset[1] * self.geo[5],
                            self.geo[4], self.geo[5]]
            else:
                in_data['T_R1'] = fid.GetRasterBand(1).ReadAsArray()
            dims = np.shape(in_data['T_R1'])
            fid = None
        except Exception as e:
            print('Error reading LST file %s: %s' % (str(self.p['T_R1']), str(e)))
            return

        # Read the image mosaic and get the LAI
        success, in_data['LAI'] = self._open_GDAL_image(
            self.p['LAI'], dims, 'Leaf Area Index', subset)
        if not success:
            return
        # Read the Vegetation Index
        success, in_data['VI'] = self._open_GDAL_image(
            self.p['VI'], dims, 'Vegetation Index', subset)
        if not success:
            return

        # Read the fractional cover data
        success, in_data['f_c'] = self._open_GDAL_image(
            self.p['f_c'], dims, 'Fractional Cover', subset)
        if not success:
            return
        # Read the Canopy Height data
        success, in_data['h_C'] = self._open_GDAL_image(
            self.p['h_C'], dims, 'Canopy Height', subset)
        if not success:
            return
        # Read the canopy width ratio
        success, in_data['w_C'] = self._open_GDAL_image(
            self.p['w_C'], dims, 'Canopy Width Ratio', subset)
        if not success:
            return
        # Read landcover
        success, in_data['landcover'] = self._open_GDAL_image(
            self.p['landcover'], dims, 'Landcover', subset)
        if not success:
            return
        # Read leaf angle distribution
        success, in_data['x_LAD'] = self._open_GDAL_image(
            self.p['x_LAD'], dims, 'Leaf Angle Distribution', subset)
        if not success:
            return
        # Read digital terrain model
        success, in_data['alt'] = self._open_GDAL_image(
            self.p['alt'], dims, 'Digital Terrain Model', subset)
        if not success:
            return

        # Read spectral properties
        success, in_data['rho_vis_C'] = self._open_GDAL_image(
            self.p['rho_vis_C'], dims, 'Leaf PAR Reflectance', subset)
        if not success:
            return
        success, in_data['tau_vis_C'] = self._open_GDAL_image(
            self.p['tau_vis_C'], dims, 'Leaf PAR Transmitance', subset)
        if not success:
            return
        success, in_data['rho_nir_C'] = self._open_GDAL_image(
            self.p['rho_nir_C'], dims, 'Leaf NIR Reflectance', subset)
        if not success:
            return
        success, in_data['tau_nir_C'] = self._open_GDAL_image(
            self.p['tau_nir_C'], dims, 'Leaf NIR Transmitance', subset)
        if not success:
            return
        success, in_data['rho_vis_S'] = self._open_GDAL_image(
            self.p['rho_vis_S'], dims, 'Soil PAR Reflectance', subset)
        if not success:
            return
        success, in_data['rho_nir_S'] = self._open_GDAL_image(
            self.p['rho_nir_S'], dims, 'Soil NIR Reflectance', subset)
        if not success:
            return
        success, in_data['emis_C'] = self._open_GDAL_image(
            self.p['emis_C'], dims, 'Leaf Emissivity', subset)
        if not success:
            return
        success, in_data['emis_S'] = self._open_GDAL_image(
            self.p['emis_S'], dims, 'Soil Emissivity', subset)
        if not success:
            return

        # Calculate illumination conditions
        success, lat = self._open_GDAL_image(self.p['lat'], dims, 'Latitude', subset)
        if not success:
            return
        success, lon = self._open_GDAL_image(self.p['lon'], dims, 'Longitude', subset)
        if not success:
            return
        success, stdlon = self._open_GDAL_image(self.p['stdlon'], dims, 'Standard Longitude', subset)
        if not success:
            return
        success, in_data['time'] = self._open_GDAL_image(self.p['time'], dims, 'Time', subset)
        if not success:
            return
        success, doy = self._open_GDAL_image(self.p['DOY'], dims, 'DOY', subset)
        if not success:
            return
        in_data['SZA'], in_data['SAA'] = met.calc_sun_angles(
            lat, lon, stdlon, doy, in_data['time'])

        del lat, lon, stdlon, doy

        # Wind speed
        success, in_data['u'] = self._open_GDAL_image(
            self.p['u'], dims, 'Wind speed', subset)
        if not success:
            return
        # Vapour pressure
        success, in_data['ea'] = self._open_GDAL_image(
            self.p['ea'], dims, 'Vapour pressure', subset)
        if not success:
            return
        # Air pressure
        success, in_data['p'] = self._open_GDAL_image(
            self.p['p'], dims, 'Pressure', subset)
        if not success:
            success, alt = self._open_GDAL_image(self.p['alt'], dims, 'Altitude', subset)
            if success:
                in_data['p'] = met.calc_pressure(alt)
            else:
                return
        success, in_data['S_dn'] = self._open_GDAL_image(
            self.p['S_dn'], dims, 'Shortwave irradiance', subset)
        if not success:
            return
        # Wind speed measurement height
        success, in_data['z_u'] = self._open_GDAL_image(
            self.p['z_u'], dims, 'Wind speed height', subset)
        if not success:
            return
        # Air temperature measurement height
        success, in_data['z_T'] = self._open_GDAL_image(
            self.p['z_T'], dims, 'Air temperature height', subset)
        if not success:
            return
        # Soil roughness
        success, in_data['z0_soil'] = self._open_GDAL_image(
            self.p['z0_soil'], dims, 'Soil Roughness', subset)
        if not success:
            return

        # Air temperature and longwave radiation
        success, in_data['T_A1'] = self._open_GDAL_image(
            self.p['T_A1'], dims, 'Air Temperature', subset)
        if not success:
            return
        success, in_data['L_dn'] = self._open_GDAL_image(
            self.p['L_dn'], dims, 'Longwave irradiance', subset)
        if not success:
            emisAtm = rad.calc_emiss_atm(in_data['ea'], in_data['T_A1'])
            in_data['L_dn'] = emisAtm * met.calc_stephan_boltzmann(in_data['T_A1'])

        # Processing mask
        if self.p['input_mask'] != '0':
            success, mask = self._open_GDAL_image(
                self.p['input_mask'], dims, 'input mask', subset)
            if not success:
                print("Please set input_mask=0 for processing the whole image.")
                return
        else:
            mask = np.ones(dims)
            mask[np.logical_or.reduce((in_data['landcover'] == res.WATER,
                                       in_data['landcover'] == res.URBAN,
                                       in_data['landcover'] == res.SNOW))] = 0

        mask[np.logical_or(~np.isfinite(in_data['VI']),
                           ~np.isfinite(in_data['T_R1']))] = 0

        # Bare soil reference ET fraction (used in Allen 2013 Eq. 5 to weight
        # the hot-pixel ETrF). Scalar 0-1 for a uniform background, or a
        # per-pixel raster path.
        if str(self.p['ETrF_bare']) != '0':
            success, in_data['ETrF_bare'] = self._open_GDAL_image(
                self.p['ETrF_bare'], dims, 'ETrF_bare', subset)
            if not success:
                print("Please set ETrF_bare=0 to assume zero ET for bare soil.")
                return
        else:
            in_data['ETrF_bare'] = np.zeros(dims)

        # Soil Heat Flux setup
        if self.G_form[0][0] == 0 or self.G_form[0][0] == 1:
            success, self.G_form[1] = self._open_GDAL_image(
                self.G_form[1], dims, 'G', subset)
            if not success:
                return
        elif self.G_form[0][0] == 2:
            self.G_form[1] = in_data['time']
        elif self.G_form[0][0] == 4:
            self.G_form[1] = (0.04, 0.10)
        elif self.G_form[0][0] == 5:
            self.G_form[1] = (0.04, 0.10)

        del in_data['time']

        # ======================================
        # Run the chosen model

        out_data = self.run_METRIC(in_data, mask)

        # ======================================
        # Save output files
        all_fields = self._get_output_structure()
        primary_fields = [field for field, save in all_fields.items() if save == S_P]
        ancillary_fields = [field for field, save in all_fields.items() if save == S_A]

        outdir = dirname(self.p['output_file'])
        if not exists(outdir):
            mkdir(outdir)

        if 'subset_output' in self.p:
            subset, geo = self._get_subset(self.p["subset_output"], self.prj, self.geo)
            dims = (subset[3], subset[2])

            for field in primary_fields:
                out_data[field] = out_data[field][subset[1]:subset[1]+subset[3],
                                                  subset[0]:subset[0]+subset[2]]
            for field in ancillary_fields:
                out_data[field] = out_data[field][subset[1]:subset[1]+subset[3],
                                                  subset[0]:subset[0]+subset[2]]

        if dims[0] <= 0 or dims[1] <= 0:
            print('No valid extent for creating output')
            return in_data, out_data

        self._write_raster_output(
            self.p['output_file'],
            out_data,
            primary_fields)

        outputfile = splitext(self.p['output_file'])[0] + '_ancillary' + \
                     splitext(self.p['output_file'])[1]
        self._write_raster_output(
            outputfile,
            out_data,
            ancillary_fields)

        print('Saved Files')

        return in_data, out_data

    def run_METRIC(self, in_data, mask=None):
        '''Execute the routines to calculate energy fluxes.

        Parameters
        ----------
        in_data : dict
            The input data for the model.
        mask : int array or None
            If None then fluxes will be calculated for all input points.

        Returns
        -------
        out_data : dict
            The output data from the model.
        '''

        print("Processing...")
        model_params = dict()

        if mask is None:
            mask = np.ones(in_data['LAI'].shape)

        dims = in_data['LAI'].shape

        # Create the output dictionary
        out_data = dict()
        all_fields = self._get_output_structure()

        for field in all_fields:
            out_data[field] = np.zeros(dims) + np.nan

        print('Estimating net shortwave radiation using Campbell two layers approach')
        # Estimate diffuse and direct irradiance
        difvis, difnir, fvis, fnir = rad.calc_difuse_ratio(
            in_data['S_dn'], in_data['SZA'], press=in_data['p'])
        out_data['fvis'] = fvis
        out_data['fnir'] = fnir
        out_data['Skyl'] = difvis * fvis + difnir * fnir
        out_data['S_dn_dir'] = in_data['S_dn'] * (1.0 - out_data['Skyl'])
        out_data['S_dn_dif'] = in_data['S_dn'] * out_data['Skyl']

        del difvis, difnir, fvis, fnir

        # ======================================
        # Net radiation for bare soil
        noVegPixels = np.logical_and(in_data['LAI'] == 0, mask == 1)
        out_data['z_0M'][noVegPixels] = in_data['z0_soil'][noVegPixels]
        out_data['d_0'][noVegPixels] = 0

        spectraGrdOSEB = out_data['fvis'] * \
            in_data['rho_vis_S'] + out_data['fnir'] * in_data['rho_nir_S']
        out_data['R_ns1'][noVegPixels] = (1. - spectraGrdOSEB[noVegPixels]) * \
            (out_data['S_dn_dir'][noVegPixels] + out_data['S_dn_dif'][noVegPixels])

        # ======================================
        # Process vegetated cases
        i = np.logical_and(in_data['LAI'] > 0, mask == 1)

        out_data['z_0M'][i], out_data['d_0'][i] = \
            res.calc_roughness(in_data['LAI'][i],
                               in_data['h_C'][i],
                               w_C=in_data['w_C'][i],
                               landcover=in_data['landcover'][i],
                               f_c=in_data['f_c'][i])

        del in_data['h_C'], in_data['w_C'], in_data['f_c']

        Sn_C1, Sn_S1 = rad.calc_Sn_Campbell(in_data['LAI'][i],
                                              in_data['SZA'][i],
                                              out_data['S_dn_dir'][i],
                                              out_data['S_dn_dif'][i],
                                              out_data['fvis'][i],
                                              out_data['fnir'][i],
                                              in_data['rho_vis_C'][i],
                                              in_data['tau_vis_C'][i],
                                              in_data['rho_nir_C'][i],
                                              in_data['tau_nir_C'][i],
                                              in_data['rho_vis_S'][i],
                                              in_data['rho_nir_S'][i],
                                              x_LAD=in_data['x_LAD'][i])

        out_data['R_ns1'][i] = Sn_C1 + Sn_S1
        del Sn_C1, Sn_S1, in_data['LAI'], in_data['x_LAD']
        del in_data['rho_vis_C'], in_data['tau_vis_C']
        del in_data['rho_nir_C'], in_data['tau_nir_C']
        del in_data['rho_vis_S'], in_data['rho_nir_S']

        out_data['emiss'] = (in_data['VI'] * in_data['emis_C']
                             + (1 - in_data['VI']) * in_data['emis_S'])

        del in_data['emis_C'], in_data['emis_S']

        out_data['albedo'] = 1.0 - out_data['R_ns1'] / in_data['S_dn']

        # No elevation delapse: UAV scenes are treated as a common-elevation
        # surface, so the moist-adiabatic datum adjustment that satellite METRIC
        # uses to normalise within-scene elevation spread would only add a
        # constant offset here (it cancels out of the dT calibration and changes
        # no per-pixel flux). Tr_datum therefore holds the raw radiometric
        # surface temperature; the name is retained to keep the downstream
        # endmember-search / calibration call sites unchanged.
        Tr_datum = in_data['T_R1']

        # Reduce potential ET based on vegetation density (Allen et al. 2013)
        out_data['ET_r_f_cold'] = np.ones(dims) * 1.05
        out_data['ET_r_f_cold'][in_data['VI'] < VI_MAX] = \
            1.05 / VI_MAX * in_data['VI'][in_data['VI'] < VI_MAX]  # Eq. 4

        out_data['ET_r_f_hot'] = (in_data['VI'] * out_data['ET_r_f_cold']
                                  + (1.0 - in_data['VI']) * in_data['ETrF_bare'])  # Eq. 5

        # ======================================
        # Resolution-adaptive CV window
        pixel_size = abs(self.geo[1])
        cv_window_m = float(self.p.get('cv_window_meters', 5.0))
        cv_win = endmember_search.compute_cv_window(pixel_size, cv_window_m)

        # Compute spatial homogeneity metrics
        cv_ndvi, _, _ = endmember_search.moving_cv_filter(in_data['VI'], cv_win)
        cv_lst, _, std_lst = endmember_search.moving_cv_filter(Tr_datum, cv_win)
        cv_albedo, _, _ = endmember_search.moving_cv_filter(out_data['albedo'], cv_win)

        # ======================================
        # Parse endmember mode configuration
        endmember_mode = str(self.p.get('endmember_mode', AUTO)).strip().lower()
        min_lst_range = float(self.p.get('min_lst_range', 3.0))

        # Create zone masks if specified
        cold_zone_mask = None
        hot_zone_mask = None

        # Priority 1: polygon feature file (endmember_zones)
        endmember_zones_path = str(self.p.get('endmember_zones', '')).strip()
        zone_field = str(self.p.get('endmember_zone_field', 'zone')).strip()
        if endmember_zones_path:
            print('Loading endmember zones from polygon file: %s'
                  % endmember_zones_path)
            cold_zone_mask, hot_zone_mask = \
                endmember_search.create_zone_masks_from_polygons(
                    dims, self.geo, self.prj,
                    endmember_zones_path, zone_field=zone_field)
            if endmember_mode == AUTO:
                endmember_mode = ZONE
                print('Endmember mode auto-promoted to "zone" '
                      '(endmember_zones file provided)')

        # Priority 2: bounding box coordinates (cold_zone, hot_zone)
        if cold_zone_mask is None and self.p.get('cold_zone', ''):
            cold_zone_str = str(self.p['cold_zone']).strip()
            if cold_zone_str:
                print('Creating cold pixel search zone mask from bounding box')
                cold_zone_mask = endmember_search.create_zone_mask(
                    dims, self.geo, cold_zone_str)

        if hot_zone_mask is None and self.p.get('hot_zone', ''):
            hot_zone_str = str(self.p['hot_zone']).strip()
            if hot_zone_str:
                print('Creating hot pixel search zone mask from bounding box')
                hot_zone_mask = endmember_search.create_zone_mask(
                    dims, self.geo, hot_zone_str)

        # Parse manual pixel coordinates
        manual_cold_global = None
        manual_hot_global = None
        if endmember_mode == MANUAL:
            manual_cold_global = self._parse_pixel_coord('cold')
            manual_hot_global = self._parse_pixel_coord('hot')
            if manual_cold_global is None or manual_hot_global is None:
                print('ERROR: manual mode requires both cold and hot pixel coordinates')
                return out_data

        # Parse prescribed dT coefficients
        prescribed_dT = None
        if endmember_mode == PRESCRIBED:
            try:
                dT_a = float(self.p['prescribed_dT_a'])
                dT_b = float(self.p['prescribed_dT_b'])
                prescribed_dT = (dT_a, dT_b)
                print('Prescribed dT mode: a=%.6f, b=%.6f' % (dT_a, dT_b))
            except (KeyError, ValueError) as e:
                print('ERROR: prescribed mode requires prescribed_dT_a and '
                      'prescribed_dT_b: %s' % str(e))
                return out_data

        # Optional hot pixel ETrF override (Allen 2013).
        # Applied at the hot pixel after endmember search; PRESCRIBED skips it
        # (no hot pixel exists when dT is supplied directly).
        hot_etrf_override = None
        hot_etrf_raw = str(self.p.get('hot_ETrF', '')).strip()
        if hot_etrf_raw:
            try:
                hot_etrf_override = float(hot_etrf_raw)
            except ValueError:
                print('WARNING: hot_ETrF="%s" is not numeric; ignoring'
                      % hot_etrf_raw)
            else:
                if endmember_mode == PRESCRIBED:
                    print('WARNING: hot_ETrF=%.3f ignored in prescribed mode '
                          '(no hot pixel)' % hot_etrf_override)
                    hot_etrf_override = None
                else:
                    print('Hot pixel ETrF override: %.3f' % hot_etrf_override)

        print('Endmember mode: %s' % endmember_mode)

        # ======================================
        # Resolve reference surface type — drives ET0 flavor and G ratio.
        ref_str = str(self.p.get('reference_type', 'tall')).strip().lower()
        if ref_str == 'tall':
            ref_flag = TALL_REFERENCE
            g_ratio = float(self.p.get('G_tall', 0.04))
        elif ref_str == 'short':
            ref_flag = SHORT_REFERENCE
            g_ratio = float(self.p.get('G_short', 0.10))
        else:
            print('WARNING: unknown reference_type "%s", defaulting to tall'
                  % ref_str)
            ref_flag = TALL_REFERENCE
            g_ratio = float(self.p.get('G_tall', 0.04))
        print('Reference surface: %s (ET0=%s, G/Rn=%.3f for G_form=4)'
              % (ref_str, 'alfalfa' if ref_flag == TALL_REFERENCE else 'grass',
                 g_ratio))

        # ======================================
        # Single-pass METRIC: process all valid pixels in one call.
        # Landcover is no longer used to split the image — it still flows
        # through res.calc_roughness for per-pixel roughness/displacement.
        aoi = mask == 1
        out_data['flag'] = np.ones(dims, dtype=np.uint8) * 255
        out_data['T_sd'] = -9999
        out_data['T_vw'] = -9999
        out_data['VI_sd'] = -9999
        out_data['VI_vw'] = -9999
        out_data['cold_pixel_global'] = -9999
        out_data['hot_pixel_global'] = -9999
        out_data['LE_cold'] = -9999
        out_data['LE_hot'] = -9999

        if not np.any(aoi):
            print('No valid pixels to process — check mask and landcover.')
            return out_data

        # Reference ET (instantaneous, energy units W/m2) over all valid pixels.
        # ETref holds the active reference ET (tall or short per reference_type)
        # as a latent-heat flux. If the user provided a station reference ET
        # (e.g. CoAgMET) matching the active reference surface, use it directly
        # for both ETref and ETref_datum (Option A: the provided value IS the
        # anchor, no datum re-scaling). Otherwise compute it internally with
        # pet_asce.
        #   reference_type=tall -> etr_inst (alfalfa);
        #   reference_type=short -> eto_inst (grass).
        if ref_flag == TALL_REFERENCE:
            provided_raw = str(self.p.get('etr_inst', '')).strip()
            provided_label = 'etr_inst (alfalfa)'
        else:
            provided_raw = str(self.p.get('eto_inst', '')).strip()
            provided_label = 'eto_inst (grass)'

        # Internal ASCE reference ET at actual conditions (W/m2). Always
        # computed: it is the fallback, and the cross-check against a provided
        # value. Note this now uses T_A1 (no datum delapse).
        internal_et0 = METRIC.pet_asce(
            in_data['T_A1'][aoi], in_data['u'][aoi], in_data['ea'][aoi],
            in_data['p'][aoi], in_data['S_dn'][aoi],
            in_data['z_u'][aoi], in_data['z_T'][aoi],
            f_cd=1, reference=ref_flag)

        if provided_raw:
            etr_mm_hr = float(provided_raw)
            # mm/hr -> W/m2 via latent heat of vaporisation at air temperature.
            lam = met.calc_lambda(in_data['T_A1'][aoi])   # J/kg
            provided_et0 = etr_mm_hr * lam / 3600.0
            out_data['ETref'][aoi] = provided_et0
            out_data['ETref_datum'][aoi] = provided_et0
            # Gross-error tripwire: compare provided vs internal ASCE estimate.
            prov_mean = float(np.nanmean(provided_et0))
            int_mean = float(np.nanmean(internal_et0))
            pct = 100.0 * abs(prov_mean - int_mean) / int_mean if int_mean else 0.0
            print('Reference ET: using provided %s = %.4f mm/hr (%.1f W/m2)'
                  % (provided_label, etr_mm_hr, prov_mean))
            print('  internal ASCE estimate = %.1f W/m2 (%.4f mm/hr); '
                  'provided vs internal differ by %.1f%%'
                  % (int_mean, int_mean * 3600.0 / float(np.nanmean(lam)), pct))
            if pct > 15.0:
                print('  WARNING: provided reference ET differs from the '
                      'internal estimate by >15%%. Check that the value matches '
                      'reference_type=%s and the flight conditions.' % ref_str)
        else:
            out_data['ETref'][aoi] = internal_et0
            out_data['ETref_datum'][aoi] = internal_et0

        # Endmember selection
        if endmember_mode == PRESCRIBED:
            out_data['cold_pixel_global'] = 'prescribed'
            out_data['hot_pixel_global'] = 'prescribed'

        elif endmember_mode == MANUAL:
            print('Using manually specified endmember pixels')
            cold_local = endmember_search.global_to_aoi_index(manual_cold_global, aoi)
            hot_local = endmember_search.global_to_aoi_index(manual_hot_global, aoi)
            if cold_local is None or hot_local is None:
                print('ERROR: manual pixel coordinates not within AOI')
                return out_data
            out_data['cold_pixel'] = cold_local
            out_data['hot_pixel'] = hot_local
            endmember_search.validate_endmembers(
                Tr_datum[aoi], in_data['VI'][aoi],
                cold_local, hot_local, min_lst_range)
            if hot_etrf_override is not None:
                out_data['ET_r_f_hot'][manual_hot_global] = hot_etrf_override
            out_data['T_sd'] = float(Tr_datum[manual_hot_global])
            out_data['T_vw'] = float(Tr_datum[manual_cold_global])
            out_data['VI_sd'] = float(in_data['VI'][manual_hot_global])
            out_data['VI_vw'] = float(in_data['VI'][manual_cold_global])
            out_data['cold_pixel_global'] = manual_cold_global
            out_data['hot_pixel_global'] = manual_hot_global
            out_data['LE_cold'] = float(
                out_data['ET_r_f_cold'][manual_cold_global]
                * out_data['ETref_datum'][manual_cold_global])
            out_data['LE_hot'] = float(
                out_data['ET_r_f_hot'][manual_hot_global]
                * out_data['ETref_datum'][manual_hot_global])

        else:
            # AUTO or ZONE — run automated search.
            # ZONE: cold_zone_mask / hot_zone_mask restrict the search.
            # AUTO: those masks are None; search the entire AOI.
            print('Automatic search of METRIC hot and cold pixels')

            if self.endmember_search == ESA:
                cold_global, hot_global = endmember_search.esa(
                    in_data['VI'], Tr_datum,
                    cv_ndvi, std_lst, cv_albedo,
                    search_mask=aoi,
                    cold_zone=cold_zone_mask, hot_zone=hot_zone_mask)
            elif self.endmember_search == MINMAX:
                cold_global, hot_global = endmember_search.maxmin_temperature(
                    in_data['VI'], Tr_datum,
                    search_mask=aoi,
                    cold_zone=cold_zone_mask, hot_zone=hot_zone_mask)
            else:  # default to CIMEC
                cold_global, hot_global = endmember_search.cimec(
                    in_data['VI'], Tr_datum,
                    out_data['albedo'], in_data['SZA'],
                    cv_ndvi, cv_lst,
                    adjust_rainfall=False,
                    search_mask=aoi,
                    cold_zone=cold_zone_mask, hot_zone=hot_zone_mask)

            if cold_global is None or hot_global is None:
                print('ERROR: endmember search failed')
                return out_data

            endmember_search.validate_endmembers(
                Tr_datum, in_data['VI'],
                cold_global, hot_global, min_lst_range)

            cold_local = endmember_search.global_to_aoi_index(cold_global, aoi)
            hot_local = endmember_search.global_to_aoi_index(hot_global, aoi)
            if cold_local is None or hot_local is None:
                print('ERROR: endmember pixels not within AOI after conversion')
                return out_data

            out_data['cold_pixel'] = cold_local
            out_data['hot_pixel'] = hot_local
            if hot_etrf_override is not None:
                out_data['ET_r_f_hot'][hot_global] = hot_etrf_override
            out_data['T_sd'] = float(Tr_datum[hot_global])
            out_data['T_vw'] = float(Tr_datum[cold_global])
            out_data['VI_sd'] = float(in_data['VI'][hot_global])
            out_data['VI_vw'] = float(in_data['VI'][cold_global])
            out_data['cold_pixel_global'] = cold_global
            out_data['hot_pixel_global'] = hot_global
            out_data['LE_cold'] = float(
                out_data['ET_r_f_cold'][cold_global]
                * out_data['ETref_datum'][cold_global])
            out_data['LE_hot'] = float(
                out_data['ET_r_f_hot'][hot_global]
                * out_data['ETref_datum'][hot_global])

        # G parameters for the AOI
        if self.G_form[0][0] == 4:
            model_params["calcG_params"] = [
                [1], np.ones(in_data['T_R1'][aoi].shape) * g_ratio]
        elif self.G_form[0][0] == 5:
            model_params["calcG_params"] = [
                [1], (in_data['T_R1'][aoi] - 273.15)
                * (0.0038 + 0.0074 * out_data['albedo'][aoi])
                * (1.0 - 0.98 * in_data['VI'][aoi] ** 4)]
        else:
            model_params["calcG_params"] = [self.G_form[0], self.G_form[1][aoi]]

        self._call_flux_model(in_data, out_data, model_params, aoi,
                              prescribed_dT=prescribed_dT)

        del model_params, aoi, Tr_datum, cv_ndvi, cv_lst, std_lst, cv_albedo

        # Calculate the global net radiation
        out_data['R_n1'] = out_data['R_ns1'] + out_data['R_nl1']
        out_data['fETr'] = out_data['LE1'] / out_data['ETref']

        print("Finished processing!")
        return out_data

    def _call_flux_model(self, in_data, out_data, model_params, i,
                         prescribed_dT=None):
        '''Call METRIC model to calculate fluxes for all data points.

        Parameters
        ----------
        in_data : dict
        out_data : dict
        model_params : dict
        i : bool array
            AOI mask
        prescribed_dT : tuple or None
            If (dT_a, dT_b), bypasses endmember calibration
        '''

        [out_data['flag'][i], out_data['R_nl1'][i], out_data['LE1'][i],
         out_data['H1'][i], out_data['G1'][i], out_data['R_A1'][i],
         out_data['u_friction'][i], out_data['L'][i],
         out_data['n_iterations'][i]] = \
            METRIC.METRIC(in_data['T_R1'][i],
                          in_data['T_A1'][i],
                          in_data['u'][i],
                          in_data['ea'][i],
                          in_data['p'][i],
                          out_data['R_ns1'][i],
                          in_data['L_dn'][i],
                          out_data['emiss'][i],
                          out_data['z_0M'][i],
                          out_data['d_0'][i],
                          in_data['z_u'][i],
                          in_data['z_T'][i],
                          out_data.get('cold_pixel'),
                          out_data.get('hot_pixel'),
                          out_data['ET_r_f_cold'][i] * out_data['ETref_datum'][i],
                          LE_hot=out_data['ET_r_f_hot'][i] * out_data['ETref_datum'][i],
                          use_METRIC_resistance=self.use_METRIC_resistance,
                          calcG_params=model_params["calcG_params"],
                          # Common-elevation UAV scenes: no DEM delapse. The
                          # solver's UseDEM=False path sets Tr_datum=Tr_K and
                          # rho_datum=rho, consistent with the no-delapse
                          # Tr_datum above.
                          UseDEM=False,
                          prescribed_dT=prescribed_dT)

    def _parse_pixel_coord(self, pixel_type):
        """Parse pixel coordinates from config (cold or hot).

        Supports both pixel (row,col) and geographic (x,y) formats.

        Parameters
        ----------
        pixel_type : str
            'cold' or 'hot'

        Returns
        -------
        global_coord : tuple (row, col) or None
        """
        # Try geographic coordinates first
        xy_key = '%s_pixel_xy' % pixel_type
        if self.p.get(xy_key, ''):
            xy_str = str(self.p[xy_key]).strip()
            if xy_str:
                x, y = [float(v.strip()) for v in xy_str.split(',')]
                row, col = endmember_search.geo_to_pixel(self.geo, x, y)
                print('%s pixel from geographic coords (%.6f, %.6f) -> '
                      'pixel (%d, %d)' % (pixel_type.capitalize(), x, y, row, col))
                return (row, col)

        # Try pixel coordinates
        rc_key = '%s_pixel_rc' % pixel_type
        if self.p.get(rc_key, ''):
            rc_str = str(self.p[rc_key]).strip()
            if rc_str:
                r, c = [int(v.strip()) for v in rc_str.split(',')]
                print('%s pixel from pixel coords: (%d, %d)'
                      % (pixel_type.capitalize(), r, c))
                return (r, c)

        print('ERROR: no %s pixel coordinates found in config' % pixel_type)
        return None

    def _open_GDAL_image(self, inputString, dims, variable, subset=[]):
        '''Open a GDAL image and returns an array with its first band.'''

        if inputString == "":
            return False, None

        success = True
        array = None
        try:
            array = np.zeros(dims) + float(inputString)
        except (ValueError, TypeError):
            try:
                fid = gdal.Open(inputString, gdal.GA_ReadOnly)
                if subset:
                    array = fid.GetRasterBand(1).ReadAsArray(
                        subset[0], subset[1], subset[2], subset[3])
                else:
                    array = fid.GetRasterBand(1).ReadAsArray()
                fid = None
            except Exception:
                print('ERROR: file read ' + str(inputString)
                      + '\n Please type a valid file name or a numeric value for '
                      + variable)
                success = False

        return success, array

    def _write_raster_output(self, outfile, output, fields):
        '''Write the specified arrays of a dictionary to a raster file.'''

        ext = splitext(outfile)[1]
        if ext.lower() == ".nc":
            driver_name = "netCDF"
            opt = []
        elif ext.lower() == ".vrt":
            driver_name = "VRT"
            opt = []
        else:
            driver_name = "GTiff"
            opt = []

        if driver_name in ["GTiff", "netCDF"]:
            rows, cols = np.shape(output['H1'])
            driver = gdal.GetDriverByName(driver_name)
            nbands = len(fields)
            ds = driver.Create(outfile, cols, rows, nbands, gdal.GDT_Float32, opt)
            ds.SetGeoTransform(self.geo)
            ds.SetProjection(self.prj)
            for i, field in enumerate(fields):
                band = ds.GetRasterBand(i + 1)
                band.SetNoDataValue(np.nan)
                band.WriteArray(output[field])
                band.FlushCache()
            ds.FlushCache()
            del ds

            if driver_name == "netCDF":
                ds = Dataset(outfile, 'a')
                grid_mapping = ds["Band1"].grid_mapping
                for i, field in enumerate(fields):
                    ds.renameVariable("Band" + str(i + 1), field)
                    ds[field].grid_mapping = grid_mapping
                ds.close()
        else:
            out_dir = join(dirname(outfile),
                           splitext(basename(outfile))[0] + ".data")
            if not exists(out_dir):
                mkdir(out_dir)
            out_files = []
            rows, cols = np.shape(output['H1'])
            for i, field in enumerate(fields):
                driver = gdal.GetDriverByName("GTiff")
                out_path = join(out_dir, field + ".tif")
                ds = driver.Create(out_path, cols, rows, 1, gdal.GDT_Float32, opt)
                ds.SetGeoTransform(self.geo)
                ds.SetProjection(self.prj)
                band = ds.GetRasterBand(1)
                band.SetNoDataValue(np.nan)
                band.WriteArray(output[field])
                band.FlushCache()
                ds.FlushCache()
                out_files.extend([out_path])

            out_vrt = out_dir.replace('.data', '.vrt')
            print(out_files)
            gdal.BuildVRT(out_vrt, out_files, separate=True)
