from astropy.io import fits
from astropy import units as u
from astropy import uncertainty as unc
from astropy.modeling import models
from pathlib import Path
import numpy as np
import scipy
import warnings
import argparse
import specutils
from specutils import Spectrum1D
from specutils.spectra import SpectralRegion
from specutils.manipulation import noise_region_uncertainty, extract_region, gaussian_smooth, trapezoid_smooth
from specutils.fitting import find_lines_threshold, estimate_line_parameters, fit_lines


# GLOBAL VARIABLES
current_dir = Path.cwd()


def read_fits(folder='TIC220414682'):
    """Function which reads all the CERES reduced FIDEOS fits image and
    returns a multi-array of the data accesible and a list with the 
    headers of each fits image.
    
    The function needs a string with the name of the target in TIC 
    format as an input (default=TIC220414682).
    """
    
    folder = current_dir / 'targets' / folder
    fitsList = [x for x in folder.iterdir() if not (x.is_dir())]
    spectra = []
    if len(fitsList) < 2:
        print(f'Target {folder} has only 1 spectrum. Skipping...\n')
        return spectra
    for fileName in fitsList:
        with fits.open(fileName) as hdul:
            data = hdul[0].data
            spectra.append(data)
    return np.array(spectra)


def shifting(spectra):
    """Function which receives a read_fits output tuple elements and
    returns the normalized spectrum for each order at RV = refRV and each
    fits file.
    """
    # We cycle through all the fits image
    reference_pos = 0.
    shifts = np.zeros(spectra.shape[0])
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        for specNum in range(spectra.shape[0]):
            linesList = []
            spectraList = []
            wls_aux = np.empty(spectra[specNum, 0, :].shape)
            flux_aux = np.empty(spectra[specNum, 5, :].shape)
            mask = np.ones(spectra[specNum, 0, :].shape)
            for order in range(spectra.shape[2]):
                wls_aux[order] = spectra[specNum, 0, order]
                flux_aux[order] = spectra[specNum, 5, order]
                # We can use the wls_aux instead of wls cuz snr is better at longer wl
                central_wavelength = np.median(wls_aux[order])  
                # Se encuentra los rangos espectrales superpuestos entre ordenes contiguos.
                if order > 0:
                    mask = wls_aux[order] < shortestWl
                    wls, flux = wls_aux[order][mask], flux_aux[order][mask]
                    spectrum = Spectrum1D(flux=(flux-np.median(flux))*u.adu, spectral_axis=wls*u.AA)
                else:
                    spectrum = Spectrum1D(flux=(flux_aux[order]-np.median(flux_aux[order]))*u.adu,
                                          spectral_axis=wls_aux[order]*u.AA)    
                spectrum = gaussian_smooth(spectrum, stddev=8)
                spectrum = noise_region_uncertainty(spectrum, SpectralRegion(5825*u.AA, 5850*u.AA))
                if not(np.any(np.isnan(spectrum.uncertainty.array))):
                    
                    lines = find_lines_threshold(spectrum, noise_factor=3)
                    linesList.append(lines[lines['line_type'] == 'absorption'])
                    spectraList.append(spectrum)
                shortestWl = wls_aux[order, 0]
            linesList = np.array(linesList)
            spectraList = np.array(spectraList)
            deep = 0  # depth of deepest line
            pos = 0   # wavelength of deepest line
            pos2 = 0  # wavelength of the 2nd deepest line
            for order in range(linesList.shape[0]):
                for i in linesList[order]['line_center_index']:
                    line_region = SpectralRegion(spectraList[order].spectral_axis[i-4], spectraList[order].spectral_axis[i+4])
                    line_spectrum = extract_region(spectraList[order], line_region)
                    estimation = estimate_line_parameters(line_spectrum, models.Gaussian1D())
                    if estimation.amplitude < deep:
                        deep = estimation.amplitude
                        pos2 = pos
                        pos = estimation.mean
            # The first spectrum is the reference spectrum, so we need to save the reference position.
            if specNum == 0:
                reference_pos = pos.value
            else:
                shifts[specNum] = pos.value - reference_pos
    # First we create the arrays which will hold the zeroed spectrum information
    fullSpectrum = np.zeros((spectra.shape[0], spectra.shape[2]), dtype=object)
    signal2Noise = np.zeros((spectra.shape[0], spectra.shape[2], 2048))
    # Now we cycle through all the fits images again and we shift them and save them.
    for specNum in range(spectra.shape[0]):
        for order in range(spectra.shape[2]):
            shifted_order = Spectrum1D(spectral_axis=spectra[specNum, 0, order]*u.AA + shifts[specNum]*u.AA,
                                       flux=spectra[specNum, 5, order]*u.adu)
            fullSpectrum[specNum, order] = shifted_order
            signal2Noise[specNum, order] = spectra[specNum, 8, order]
    return fullSpectrum, signal2Noise


def coadding(fullSpectrum, signal2Noise, name='TIC220414682'):
    """Function which receives more than 1 zeroed spectrum and coadds 
    them with a weighted average, with the weights being the SNR of the 
    resolution element.This function receives a spectrum output from 
    the function "zeroing". This function returns one spectrum data 
    file with shape (2, 2048).
    """

    # First we need to allocate memory for the output
    coaddedSpectrum = np.zeros((fullSpectrum.shape[1], 2, 2048))
    # Then we need to create a new array which will hold the wavelength
    # and flux values for ONE order for ALL the fits.
    for order in range(fullSpectrum.shape[1]):
        orderData = np.zeros((fullSpectrum.shape[0], 2, 2048))
        wls = np.zeros(2048)
        for fitsNum in range(fullSpectrum.shape[0]):
            # In this function index 0 is flux and index 1 is 
            # wavelength
            orderData[fitsNum, 0, :] = fullSpectrum[fitsNum, order].flux.value
            orderData[fitsNum, 1, :] = fullSpectrum[fitsNum, order].spectral_axis.value
            # Now we use the first spectrum wavelength axis as the domain which will be used for the
            # interpolation of the other files and the final coadded spectrum.
            if fitsNum > 0:
                spl = scipy.interpolate.splrep(orderData[fitsNum, 1, :], orderData[fitsNum, 0, :], k=3)
                splFlux = scipy.interpolate.splev(wls, spl)
                orderData[fitsNum, 0, :] = splFlux
            else:
                wls = orderData[0, 1, :]
        # Now we coadd the spectra for the order, considering SNR as weights
        if np.any(np.sum(signal2Noise[:, order, :], axis=0) == 0.0):
            warnings.warn(f'SNR in order {order} as weights sum to 0 for target {name}. Using a non weighted average')
            coaddedSpectrum[order, 0, :] = np.average(orderData[:, 0, :], axis=0)
        else:
            coaddedSpectrum[order, 0, :] = np.average(orderData[:, 0, :], axis=0, weights=signal2Noise[:, order, :])
        coaddedSpectrum[order, 1, :] = wls
    return coaddedSpectrum


def data_to_ceres(coadded, name='TIC220414682'):
    """Function which receives coadded spectrum from coadding function and 
    a target name (default=TIC220414682) and returns a data array with
    shape appropiate for CERES pipeline. It also writes a fits file in
    a subfolder named "coadded".
    """
    folder = current_dir / 'targets' / name
    fitsList = [x for x in folder.iterdir() if not (x.is_dir())]
    fileName = fitsList[0]
    with fits.open(fileName) as hdul:
        data = hdul[0].data
    # We only overwrite the following two "types" because those are used by the CERES
    # pipeline for computing star params.
    #   data[type, order, values]   coadded[order, type, values]
    data[0, :, :] = coadded[:, 1, :]  # Overwrite wavelength
    data[5, :, :] = coadded[:, 0, :]  # Overwrite normalized flux
    hdu = fits.PrimaryHDU(data)
    Path(folder / 'coadded').mkdir(parents=True, exist_ok=True)
    outputPath = folder / 'coadded' / f'{len(fitsList)}stack_{name}.fits'
    hdu.writeto(outputPath, overwrite=True)
    print(f'Output fits file saved in {outputPath} .\n')
    return


def main(targetList=None):
    if targetList is None:
        raise Exception('Program needs a relative path to the target list to stack spectra.')
    path = current_dir / targetList
    with open(path, 'r') as f:
        lines = f.readlines()
        for target in lines:
            spectra= read_fits(folder=target.strip())
            if len(spectra) < 2:
                continue
            fullSpectrum, signal2Noise = shifting(spectra)
            coadded = coadding(fullSpectrum, signal2Noise, target.strip())
            print(f'Target {target.strip()} has been coadded with {fullSpectrum.shape[0]} spectra.')
            data_to_ceres(coadded, name=target.strip())
    with open(current_dir / "targets" / "stackedspectra.txt", "w") as file:
        pass
    with open(path, "r") as f:
        lines = f.readlines()
        for target in lines:
            folder = current_dir / 'targets' / target.strip()
            fitsList = [x for x in folder.iterdir() if not (x.is_dir())]
            with open(current_dir / "targets" / "stackedspectra.txt", "a") as file:
                file.write(f"{len(fitsList)}stack_{target.strip()}\n")
    print("stackedspectra.txt created in targets folder.\n")
    print('Process finished.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("targetlist",
                        help="relative path to target list to stack spectra. Example: relative/path/to/targetList.txt")
    args = parser.parse_args()
    main(targetList=args.targetlist)