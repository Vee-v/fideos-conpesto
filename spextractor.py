from astropy.io import fits
from astropy import units as u
from pathlib import Path
import numpy as np
import scipy
import warnings
import argparse
import specutils
from specutils import Spectrum1D

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
    headers = []
    if len(fitsList) < 2:
        print(f'Target {folder} has only 1 spectrum. Skipping...\n')
        return spectra, headers
    for fileName in fitsList:
        with fits.open(fileName) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data
            spectra.append(data)
            headers.append(hdr)
    return np.array(spectra), headers


def zeroing(spectra, headers):
    """Function which receives a read_fits output tuple elements and
    returns the normalized spectrum for each order at RV = 0 and each
    fits file.
    """

    # First we create the arrays which will hold the zeroed spectrum
    # information:
    fullSpectrum = np.zeros((spectra.shape[0], spectra.shape[2]), dtype=object)
    signal2Noise = np.zeros((spectra.shape[0], spectra.shape[2], 2048))
    # Then we cycle through all the fits image data and header.
    for fitsNum in range(spectra.shape[0]):
        # We allocate the RV, Flux array and Wavelength array for each fits.
        rv = headers[fitsNum]['RV'] * u.Unit("km/s")
        flux = spectra[fitsNum][5] * u.adu
        wl = spectra[fitsNum][0] * u.AA
        snr = spectra[fitsNum][8]  # We don't need to shift the SNR because the operations are index-based.
        # We create the 1D spectrum object from specutils package for each order
        for order in range(spectra.shape[2]):
            spectrum = Spectrum1D(
                spectral_axis=wl[order],
                flux=flux[order],
                radial_velocity=rv
            )
            # Now we shift the spectrum to RV=0.
            try:
                spectrum.shift_spectrum_to(radial_velocity=0 * u.Unit("km/s"))
            except:
                raise Exception(
                    f'This function needs specutils version >=1.8.0. Your version is {specutils.__version__}.')
            fullSpectrum[fitsNum][order] = spectrum
            signal2Noise[fitsNum, order, :] = snr[order]
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
            spectra, header = read_fits(folder=target.strip())
            if len(spectra) < 2:
                continue
            fullSpectrum, signal2Noise = zeroing(spectra, header)
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
