
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from reproject import reproject_interp
from regions import Regions
from spectral_cube import SpectralCube
from lmfit.models import GaussianModel
from numpy import exp, sqrt, pi

def setup_plotting():
    '''Configure plotting properties.'''
    from matplotlib import rc
    rc('font', **{'family':'serif', 'serif':['serif']})
    rc('text', usetex=True)
    plt.rcParams.update({'font.size': 15})

def read_spectral_cube(file_path):
    '''Reads a spectral cube from a FITS file.'''
    return SpectralCube.read(file_path)

def process_spectral_regions(cube, regions_file):
    '''Extracts intensity and velocity data for defined regions in a spectral cube.'''
    region_list = Regions.read(regions_file, format='ds9')
    intensities = []
    velocities = []
    for region in region_list:
        sub_cube = cube.subcube_from_regions([region])
        spectrum = sub_cube.mean(axis=(1, 2))
        intensities.append(spectrum.to_value())
        velocities.append(spectrum.spectral_axis.to_value('u.km/u.s'))
    return intensities, velocities, region_list

def gaussian_model():
    '''Builds a Gaussian model for curve fitting.'''
    model = GaussianModel(prefix='g1_') + GaussianModel(prefix='g2_') + GaussianModel(prefix='g3_') + GaussianModel(prefix='g4_')
    params = model.make_params(
        g1_amplitude=5, g1_center=-65, g1_sigma=0.1,
        g2_amplitude=5, g2_center=-58, g2_sigma=0.1,
        g3_amplitude=5, g3_center=-10.4, g3_sigma=0.1,
        g4_amplitude=5, g4_center=-48, g4_sigma=0.1
    )
    return model, params

def plot_spectral_data(velocity, intensity, labels):
    '''Plots spectral data.'''
    fig, ax = plt.subplots(3, 5, sharex='col', sharey='row', figsize=(15, 10))
    fig.subplots_adjust(hspace=0, wspace=0)
    for i in range(15):
        ax = plt.subplot(3, 5, i + 1)
        ax.step(velocity[i], intensity[i], color='black', linewidth=1, label=labels[i])
        ax.set_xlim(-80, -30)
        ax.annotate(labels[i], xy=(0.05, 0.85), xycoords='axes fraction', fontsize=14)
        if i == 10:
            ax.set_xlabel(r'v$_\mathrm{LSR}$ [km s$^{-1}$]')
            ax.set_ylabel(r'T$_\mathrm{mb}$ [K]')
    plt.savefig('NGC7538_CII_CO32_fitted.pdf', dpi=400)
    plt.show()

def main():
    setup_plotting()
    cii_cube_path = '/path/to/NGC7538_CII_merged_fullmap.fits'
    regions_file = 'Positions4SpectralAnalysis_28July2023.reg'

    cii_cube = read_spectral_cube(cii_cube_path)
    intensity_cii, velocity_cii, region_list = process_spectral_regions(cii_cube, regions_file)

    model, params = gaussian_model()
    result = model.fit(intensity_cii[0], params, x=velocity_cii[0])

    print(result.fit_report())

    labels = ['Region {}'.format(i+1) for i in range(15)]
    plot_spectral_data(velocity_cii, intensity_cii, labels)

if __name__ == '__main__':
    main()
