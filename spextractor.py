from astropy.io import fits
from astropy import units as u
from astropy import uncertainty as unc
from astropy.nddata import StdDevUncertainty
from pathlib import Path
import numpy as np
import scipy
import warnings
import argparse
import specutils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from specutils import Spectrum1D
from specutils.manipulation import LinearInterpolatedResampler
from specutils.analysis import template_correlate

# GLOBAL VARIABLES
current_dir = Path.cwd()

def gaussian(x, mu, sig, A):
    return A * np.exp(- .5 * np.power(x - mu, 2) / np.power(sig, 2))# + b * np.power(x, 2) + c * x + d


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


def ccf(spectra, target, pdf_output):
    """Function which receives the CERES reduced spectra of a star and calculates the cross correlation function
    between the first spectrum and each of the other spectra in wavelength log space to estimate the wavelength
    displacement due to radial velocities.
    """
    # We cycle through all the fits image
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        noOverlapSpectra = []
        for specNum in range(spectra.shape[0]):
            spectraList = []
            wls_aux = np.empty(spectra[specNum, 0, :].shape)
            flux_aux = np.empty(spectra[specNum, 5, :].shape)
            mask = np.ones(spectra[specNum, 0, :].shape)
            # In this section we mask the overlapped wavelengths of each order.
            for order in range(spectra.shape[2]):
                wls_aux[order] = spectra[specNum, 0, order]
                flux_aux[order] = spectra[specNum, 5, order]
                # We can use the wls_aux instead of wls cuz snr is better at longer wl
                central_wavelength = np.median(wls_aux[order])  
                # Se encuentra los rangos espectrales superpuestos entre ordenes contiguos.
                if order > 0:
                    mask = wls_aux[order] < shortestWl
                    wls, flux = wls_aux[order][mask], flux_aux[order][mask]
                    uncertainty = StdDevUncertainty(0.1*np.ones(wls.shape[0])*u.adu)
                    rest_value = 6000. * u.AA  # not sure the implicancies of this
                    spectrum = Spectrum1D(flux=(flux-np.median(flux))*u.adu, spectral_axis=wls*u.AA,
                                          uncertainty=uncertainty, velocity_convention='optical',
                                          rest_value=rest_value)
                else:
                    uncertainty = StdDevUncertainty(0.1*np.ones(wls_aux[order].shape[0])*u.adu)
                    spectrum = Spectrum1D(flux=(flux_aux[order]-np.median(flux_aux[order]))*u.adu,
                                          spectral_axis=wls_aux[order]*u.AA, uncertainty=uncertainty)    
                spectraList.append(spectrum)
                shortestWl = wls_aux[order, 0]
            noOverlapSpectra.append(spectraList)
        # Here we calculate the CCF for each spectrum against the first taken.
        figs = []
        RV_shifts = []
        for specNum in range(0, spectra.shape[0]):
            corrCCF = []
            lagsCCF =[]
            for order in range(8,26):
                corr, lags = template_correlate(noOverlapSpectra[specNum][order], noOverlapSpectra[0][order])
                corrCCF.append(corr)
                lagsCCF.append(lags)
            # Here we resample for averaging the CCF
            orderCCF =[]
            for i, j in enumerate(corrCCF):
                wav = lagsCCF[i].value
                spectrum = Spectrum1D(flux=j*u.adu, spectral_axis=wav * u.AA)
                orderCCF.append(spectrum)
            resample_grid = lagsCCF[-1].value
            resample_grid *= u.AA
            ccf_resample = LinearInterpolatedResampler()
            outputCCF = np.zeros((18, len(resample_grid)))
            x = 0
            for i in orderCCF:
                output_ccf = ccf_resample(i, resample_grid)
                outputCCF[x, :] = output_ccf.flux.value
                x += 1
            avgCCF = np.average(outputCCF, axis=0)
            fig = plt.figure()
            for i in range(18):
                plt.plot(lagsCCF[i], corrCCF[i], alpha=0.1)
            plt.plot(lagsCCF[-1], avgCCF, color='black', label='Average CCF')
            popt, pcov = scipy.optimize.curve_fit(gaussian, lagsCCF[-1], avgCCF)
            RV_shifts.append(popt[0] *u.km/u.s)
            plt.plot(lagsCCF[-1], gaussian(lagsCCF[-1].value, *popt), ls='-.', color='r', label='Gaussian Fit')
            plt.xlim(-100,100)
            plt.legend()
            plt.axvline(popt[0], color='black', ls='--')
            plt.title(f'$\Delta RV={popt[0]:.5f}$km/s | {target.strip()} | spectrum {specNum}')
            plt.ylabel('CCF power')
            plt.xlabel(r'$\Delta RV$ (km/s)')
#             plt.show()
            figs.append(fig)
        if pdf_output:
            outputname = f'{target.strip()}_{spectra.shape[0]}stack.pdf'
            p = PdfPages(outputname)
            for figure in figs:
                figure.savefig(p, format='pdf')
            p.close()
            print(f'Output pdf file saved in {current_dir/outputname}.\n')
    return RV_shifts


def shifting(spectra, shifts):
    """Function which shifts the spectra depending on the results of function ccf.
    """
        
    # First we create the arrays which will hold the zeroed spectrum information
    fullSpectrum = np.zeros((spectra.shape[0], spectra.shape[2]), dtype=object)
    signal2Noise = np.zeros((spectra.shape[0], spectra.shape[2], 2048))
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        plt.figure()
        for specNum in range(spectra.shape[0]):
            spectraList = []
            wls = np.empty(spectra[specNum, 0, :].shape)
            flux = np.empty(spectra[specNum, 5, :].shape)
            for order in range(spectra.shape[2]):
                wls[order] = spectra[specNum, 0, order]
                flux[order] = spectra[specNum, 5, order]
                # We can use the wls_aux instead of wls cuz snr is better at longer wl
                central_wavelength = np.median(wls[order])  
                # Se encuentra los rangos espectrales superpuestos entre ordenes contiguos.
                spectrum = Spectrum1D(flux=(flux[order]-np.median(flux[order]))*u.adu,
                                      spectral_axis=wls[order]*u.AA, radial_velocity=0*u.km/u.s,
                                      velocity_convention='optical')
                if order != 0:
                    spectrum.shift_spectrum_to(radial_velocity=shifts[specNum])
                fullSpectrum[specNum, order] = spectrum
                signal2Noise[specNum, order] = spectra[specNum, 8, order]
            plt.plot(fullSpectrum[specNum, 14].spectral_axis, fullSpectrum[specNum, 14].flux - specNum * u.adu)
        plt.xlim(5685, 5695)
#         plt.show()
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
    plt.plot(coaddedSpectrum[14, 1, :],coaddedSpectrum[14, 0, :])
    plt.xlim(5685, 5695)
#     plt.show()
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


def main(targetList='targetList.txt', pdf=False):
    if targetList is None:
        raise Exception('Program needs a relative path to the target list to stack spectra.')
    path = current_dir / targetList
    with open(path, 'r') as f:
        lines = f.readlines()
        for target in lines:
            spectra= read_fits(folder=target.strip())
            if len(spectra) < 2:
                continue
            shifts = ccf(spectra, target, pdf)
            fullSpectrum, signal2Noise = shifting(spectra, shifts)
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
    parser.add_argument('--pdf', action=argparse.BooleanOptionalAction, help='outputs pdf file for performance evaluation of the code')
    parser.add_argument("targetlist",
                        help="relative path to target list to stack spectra. Example: relative/path/to/targetList.txt")
    args = parser.parse_args()
    main(targetList=args.targetlist, pdf=args.pdf)
