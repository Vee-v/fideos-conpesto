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


def snr(wls, signal, orden):
    if orden == 0:
        rango = 7043, 7054
    elif orden == 1:
        rango = 6865.1, 6869
    elif orden == 2:
        rango = 6817.7, 6822
    elif orden == 3:
        rango = 6681, 6693.6
    elif orden == 4:
        rango = 6589, 6593
    elif orden == 5:
        rango = 6504.2, 6510
    elif orden == 6:
        rango = 6383.2, 6386.3
    elif orden == 7:
        rango = 6309.2, 6311.5
    elif orden == 8:
        rango = 6207,6212
    elif orden == 9:
        rango = 6068.5, 6080
    elif orden == 10:
        rango = 6048.5, 6055
    elif orden == 11:
        rango = 5969, 5975
    elif orden == 12:
        rango = 5820, 5848
    elif orden == 13:
        rango = 5768.5, 5773.5
    elif orden == 14:
        rango = 5720, 5728
    elif orden == 15:
        rango = 5609.6, 5616    
    elif orden == 16:
        rango = 5549.2, 5555
    elif orden == 17:
        rango = 5449.7, 5456.7
    elif orden == 18:
        rango = 5439.5, 5442.5
    elif orden == 19:
        rango = 5356, 5363
    elif orden == 20:
        rango = 5290.8, 5294
    elif orden == 21:
        rango = 5246, 5248
    elif orden == 22:
        rango = 5162, 5163.28
    elif orden == 23 or orden == 24:
        rango = 5047.3, 5049
    elif orden == 25:
        rango = 4944.5, 4946.8
    elif orden == 26:
        rango = 4894.8, 4897.8
    elif orden == 27:
        rango = 4851.5, 4852.8
    elif orden == 28:
        rango = 4794.37, 4797.37
    elif orden == 29:
        rango = 4720.6, 4721.95
    elif orden == 30:
        rango = 4677, 4679.5
    elif orden == 31:
        rango = 4633.5, 4632
    elif orden == 32:
        rango = 4563.1, 4565
    elif orden == 33:
        rango = 4537.78, 4539.78
    elif orden == 34:
        rango = 4487.5, 4489.3
    elif orden == 35:
        rango = 4449.5, 4450.4
    elif orden == 36:
        rango = 4412.7, 4416
    elif orden == 37:
        rango = 4358.2, 4359.7
    elif orden == 38:
        rango = 4317, 4319.5
#     plt.plot(wls, signal)
#     for line in rango:
#         plt.axvline(line)
#     plt.show()
    spectrum = Spectrum1D(flux=signal*u.adu, spectral_axis=wls*u.AA)
    noise_range = spectrum[rango[0]*u.AA:rango[1]*u.AA]
    rms = np.sqrt(np.nanmean(np.power(noise_range.flux - np.nanmedian(noise_range.flux), 2)))
    snr = spectrum.flux/rms
#     print(snr)
    return snr


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
                # Se encuentra los rangos espectrales superpuestos entre ordenes contiguos.
                spectrum = Spectrum1D(flux=(flux[order]-np.median(flux[order]))*u.adu,
                                      spectral_axis=wls[order]*u.AA, radial_velocity=0*u.km/u.s,
                                      velocity_convention='optical')
                if order != 0:
                    spectrum.shift_spectrum_to(radial_velocity=shifts[specNum])
                fullSpectrum[specNum, order] = spectrum
            plt.plot(fullSpectrum[specNum, 13].spectral_axis, fullSpectrum[specNum, 13].flux - specNum * 0.4 * u.adu)
        plt.xlim(5770, 5790)
        plt.ylim(-spectra.shape[0]*0.8, 1)
        plt.xlabel('Wavelength [Angstrom]')
        plt.ylabel('Flux [adu] + constant')
        plt.show()
    return fullSpectrum


def coadding(fullSpectrum, signal2Noise, name='TIC220414682'):
    """Function which receives more than 1 zeroed spectrum and coadds 
    them with a weighted average, with the weights being the SNR of the 
    resolution element. This function returns one spectrum data 
    file with shape (order number, 2, 2048). TODO: accept different
    numbers of resolution elements per order (now 2048).
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
    # In this section we mask the overlapped wavelengths of each order.
    ordersNum = coaddedSpectrum.shape[0]
    noOverlapCoaddedSpectrum = []
    for order in range(ordersNum):
        wls_aux  = coaddedSpectrum[order, 1, :]
        flux_aux = coaddedSpectrum[order, 0, :]
        # Se encuentra los rangos espectrales superpuestos entre ordenes contiguos.
#         if False:
        if order > 0:
            mask = wls_aux < shortestWl
            wls  = wls_aux[mask]
            flux = flux_aux[mask]
            uncertainty = StdDevUncertainty(0.1*np.ones(wls.size)*u.adu)
            rest_value = 6000. * u.AA  # not sure the implicancies of this
            spectrum = Spectrum1D(flux=(flux-np.median(flux))*u.adu, spectral_axis=wls*u.AA,
                                  uncertainty=uncertainty, velocity_convention='optical',
                                  rest_value=rest_value)
        else:
            uncertainty = StdDevUncertainty(0.1*np.ones(wls_aux.size)*u.adu)
            spectrum = Spectrum1D(flux=(flux_aux-np.median(flux_aux))*u.adu,
                                  spectral_axis=wls_aux*u.AA, uncertainty=uncertainty)    
        noOverlapCoaddedSpectrum.append(spectrum)
        shortestWl = wls_aux[0]
    plt.figure()
    plt.plot(noOverlapCoaddedSpectrum[13].spectral_axis, noOverlapCoaddedSpectrum[13].flux)
    plt.xlim(5770, 5790)
    plt.xlabel('Wavelength [Angstrom]')
    plt.ylabel('Flux [adu] + constant')
    plt.ylim(-fullSpectrum.shape[0]*0.8, 1)
    plt.show()
    noOverlapCoaddedSpectrum.reverse()
    return coaddedSpectrum, noOverlapCoaddedSpectrum


def data_to_zaspe(coadded, name='TIC220414682'):
    """Function which receives coadded spectrum and a target name
    (default is TIC220414682) and returns a data array with shape
    appropiate for input in ZASPE.
    """
    ############# THIS DOESNT WORK RIGHT NOW FOR ZASPE, USE CERES OUTPUT
    
    datasize = len(coadded[-1].spectral_axis)
    for order in range(1, len(coadded)):
        pixelsNum = len(coadded[order].spectral_axis)
        if pixelsNum < datasize:
            datasize = pixelsNum
    notFreeSpectralRange = 100 # pixels
    ones = np.ones(datasize + notFreeSpectralRange, dtype=bool)
    folder = current_dir / 'targets' / name
    with open(folder / f'{name}.dat', 'w') as file:
        pass 
    with open(folder/ f'{name}.dat', 'a') as file:
        for order in range(len(coadded)):
            if order == len(coadded) - 1:
                zeros = np.zeros(0, dtype=bool)
            else:
                zeros = np.zeros(len(coadded[order].spectral_axis) - datasize - notFreeSpectralRange, dtype=bool)
            mask = np.concatenate([ones, zeros])
            wl = coadded[order].spectral_axis.value
            wl = wl[mask]
            flux = coadded[order].flux.value+10
            flux = flux[mask]
            for pixel in range(len(flux)):
                data = np.empty(3, dtype='U24')
                data[0] = f'{order:#.18e}'
                data[1] = f'{wl[pixel]:#.18e}'
                data[2] = f'{flux[pixel]:#.18e}'
                file.write(' '.join(data) + '\n')
    print(f'Created ZASPE input file at {folder/ name}.dat')
    return


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
    data[3, :, :] = np.multiply(data[7, :, :], coadded[:, 0, :]+1)  # Overwrite deblazed flux
    plt.figure()
    plt.plot(coadded[13, 1, :], coadded[13, 0, :])
    plt.plot(coadded[13, 1, :], data[7, 13, :])
    plt.plot(coadded[13, 1, :], data[3, 13, :])
    plt.show()
    data[5, :, :] = coadded[:, 0, :]  # Overwrite norm flux
    data[0, :, :] = coadded[:, 1, :]  # Overwrite wavelength
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
            spectra = read_fits(folder=target.strip())
            if len(spectra) < 2:
                continue
            snr_array = np.zeros([spectra.shape[0], spectra.shape[2], spectra.shape[-1]])
            for spec in range(spectra.shape[0]):
                print(f'Calculating SNR of spectrum number {spec} of {target.strip()}...\n')
                for orden in range(spectra.shape[2]):
                    snr_array[spec, orden, :] = snr(spectra[spec, 0, orden, :], spectra[spec, 5, orden, :], orden)
                print(f'\tOrder 13 (573nm ~ 583nm) Median SNR of {target.strip()} for spectrum number {spec} is:\n\t\t{np.nanmedian(snr_array[spec, 13, :]):.1f}\n')
            shifts = ccf(spectra, target, pdf)
            fullSpectrum = shifting(spectra, shifts)
            coadded, noOverlapCoadded = coadding(fullSpectrum, snr_array, target.strip())
            snr_array = np.zeros([coadded.shape[0], coadded.shape[-1]])
            for orden in range(coadded.shape[0]):
                snr_array[orden, :] = snr(coadded[orden, 1, :], coadded[orden, 0, :]+1, orden)
            print(f'Target {target.strip()} has been coadded with {fullSpectrum.shape[0]} spectra.')
            print(f'\tOrder 13 (573nm ~ 583nm) Median SNR of coadded spectrum is:\n\t\t{np.nanmedian(snr_array[13, :]):.1f}\n')
#             data_to_zaspe(noOverlapCoadded, name=target.strip())
            data_to_ceres(coadded, name=target.strip())
#     with open(current_dir / "targets" / "stackedspectra.txt", "w") as file:
#         pass
#     with open(path, "r") as f:
#         lines = f.readlines()
#         for target in lines:
#             folder = current_dir / 'targets' / target.strip()
#             fitsList = [x for x in folder.iterdir() if not (x.is_dir())]
#             with open(current_dir / "targets" / "stackedspectra.txt", "a") as file:
#                 file.write(f"{len(fitsList)}stack_{target.strip()}\n")
#     print("stackedspectra.txt created in targets folder.\n")
    print('Process finished.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', action=argparse.BooleanOptionalAction, help='outputs pdf file for performance evaluation of the code')
    parser.add_argument("targetlist",
                        help="relative path to target list to stack spectra. Example: relative/path/to/targetList.txt")
    args = parser.parse_args()
    main(targetList=args.targetlist, pdf=args.pdf)