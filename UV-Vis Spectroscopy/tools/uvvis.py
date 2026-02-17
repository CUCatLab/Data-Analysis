import numpy as np
from scipy import integrate
import pandas as pd
from pandas import DataFrame as df
import yaml
import matplotlib.pyplot as plt
from lmfit.models import SkewedGaussianModel
import ipywidgets as ipw
from ipywidgets import Layout
from IPython.display import clear_output, display, Markdown as md
import os
from pathlib import Path
import io
# New for .gxz debug + parsing
import gzip
import re
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
from scipy.optimize import curve_fit

settingsFile = 'tools/settings.yaml'


class dataTools :

    def __init__(self) :
        
        pass

    def loadData(self,file) :
        
        for file_info in file :
            filename = file_info['name']
            content = file_info['content']
            try:
                if filename.lower().endswith('.csv'):
                    data = pd.read_csv(io.BytesIO(content))
                elif filename.lower().endswith('.xlsx'):
                    data = pd.read_excel(io.BytesIO(content))
                else:
                    print(filename+": Unsupported file type.")
                    data = None
                    return
                data = data.apply(pd.to_numeric, errors='coerce')
                data.columns = ["Wavelength (nm)","Absorbance (au)"]
                data = data.iloc[::-1].reset_index(drop=True)
                data = data.dropna()
                # print(filename+": loaded successfully.")
            except Exception as e:
                print(f"{filename}: Error reading file: {e}")
                data = None
        return data
    
    def plot(self, data, Title=''):
        
        fontsize = 16
        fig, ax = plt.subplots(figsize=(10, 8))
        handles = []
        labels = []
        for name in data.columns:
            if name != 'Wavelength (nm)':
                ln, = ax.plot(data["Wavelength (nm)"], data[name], label=name, lw=1.5)  # returns a Line2D
                handles.append(ln)
                labels.append(name)
        # ax.legend(handles, labels, frameon=False, bbox_to_anchor=(1.02, 1), fontsize=fontsize*0.75)
        ax.set_xlabel('Wavelength (nm)', fontsize=fontsize)
        ax.set_ylabel('Absorbance (au)', fontsize=fontsize)
        ax.set_title(Title, fontsize=fontsize)
        ax.tick_params(axis='both', which='both', labelsize=fontsize, direction="in")
        ax.minorticks_on()
        fig.tight_layout()
        plt.show()
        return fig


class analysisTools :

    def __init__(self) :

        pass

    def gaussian(self, x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def lorentzian(self, x, amp, mu, gamma):
        return amp * (gamma**2 / ((x - mu)**2 + gamma**2))

    def poly_baseline(self, x, *b):
        # b = (b0, b1, ..., bN), baseline = b0 + b1*x + ... + bN*x^N
        y = np.zeros_like(x, dtype=float)
        for k, bk in enumerate(b):
            y += bk * x**k
        return y

    def make_model(self, peak_model="gaussian", baseline_degree=2):
        baseline_degree = int(baseline_degree)
        if baseline_degree < 0:
            raise ValueError("baseline_degree must be >= 0")

        if peak_model.lower() == "gaussian":
            def model(x, amp, mu, width, *b):
                return self.gaussian(x, amp, mu, width) + self.poly_baseline(x, *b)
            width_name = "sigma"
            fwhm_from_width = lambda w: 2.35482 * w
        elif peak_model.lower() == "lorentzian":
            def model(x, amp, mu, width, *b):
                return self.lorentzian(x, amp, mu, width) + self.poly_baseline(x, *b)
            width_name = "gamma"
            fwhm_from_width = lambda w: 2.0 * w
        else:
            raise ValueError("peak_model must be 'gaussian' or 'lorentzian'")

        n_base = baseline_degree + 1  # b0..bN
        return model, n_base, width_name, fwhm_from_width

    def fit_uvvis_lambdamax(
        self,
        df,
        xcol="Wavelength (nm)",
        ycol="Absorbance (au)",
        fit_range=(200, 800),
        baseline_degree=2,
        peak_model="gaussian",
        plot=True
    ):
        # --- extract and clean ---
        data = df[[xcol, ycol]].dropna()
        x = data[xcol].to_numpy(float)
        y = data[ycol].to_numpy(float)

        # sort
        idx = np.argsort(x)
        x, y = x[idx], y[idx]

        # subset (your range 200–800 nm)
        lo, hi = fit_range
        m = (x >= lo) & (x <= hi)
        x, y = x[m], y[m]
        if x.size < 10:
            raise ValueError("Not enough points in fit_range to perform a stable fit.")

        # --- build model based on baseline degree ---
        model, n_base, width_name, fwhm_from_width = self.make_model(peak_model, baseline_degree)

        # --- initial baseline guess from edges ---
        n = x.size
        k = max(5, int(0.08 * n))  # use ~8% points at each edge
        edge = np.r_[0:k, n-k:n]
        xe, ye = x[edge], y[edge]

        # polyfit returns highest power first; convert to b0..bN
        coef = np.polyfit(xe, ye, deg=baseline_degree)  # length = baseline_degree+1
        b_init = coef[::-1]  # reverse -> b0..bN

        # baseline-subtracted for peak init
        y_base0 = self.poly_baseline(x, *b_init)
        y_sub = y - y_base0

        # --- peak init ---
        mu0 = x[np.argmax(y_sub)]
        amp0 = max(1e-6, float(np.max(y_sub)))

        # width guess via rough FWHM on baseline-subtracted signal
        half = 0.5 * amp0
        above = y_sub >= half
        if np.any(above):
            x_above = x[above]
            fwhm0 = (x_above.max() - x_above.min()) if x_above.size > 1 else 20.0
        else:
            fwhm0 = 30.0  # fallback in nm

        if peak_model.lower() == "gaussian":
            width0 = max(1e-6, fwhm0 / 2.35482)  # sigma
        else:
            width0 = max(1e-6, fwhm0 / 2.0)      # gamma

        # --- pack p0 and bounds (STRICT inequalities!) ---
        p0 = np.r_[ [amp0, mu0, width0], b_init ]

        # bounds for amp, mu, width
        lb = [0.0, x.min(), 1e-6]
        ub = [np.inf, x.max(), (x.max() - x.min())]

        # bounds for baseline coefficients: wide but finite to help stability
        # (you can loosen/tighten as desired)
        for _ in range(n_base):
            lb.append(-np.inf)
            ub.append(np.inf)

        lb = np.array(lb, float)
        ub = np.array(ub, float)

        # --- fit ---
        popt, pcov = curve_fit(
            model, x, y,
            p0=p0,
            bounds=(lb, ub),
            maxfev=50000
        )

        amp, mu, width = popt[0], popt[1], popt[2]
        b_fit = popt[3:]

        # components and metrics
        y_fit = model(x, *popt)
        y_base = self.poly_baseline(x, *b_fit)
        y_peak = y_fit - y_base

        resid = y - y_fit
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

        perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.full_like(popt, np.nan)
        mu_err = perr[1]

        result = {
            "lambda_max_nm": float(mu),
            "lambda_max_err_nm": float(mu_err),
            "amp": float(amp),
            width_name: float(width),
            "fwhm_nm": float(fwhm_from_width(width)),
            "baseline_coeffs_b0_to_bN": [float(v) for v in b_fit],
            "r2": float(r2),
            "popt": popt,
            "pcov": pcov
        }

        if plot:
            plt.figure(figsize=(8,5))
            plt.plot(x, y, "k.", ms=3, label="data")
            plt.plot(x, y_fit, "r-", lw=2, label="fit (peak+baseline)")
            plt.plot(x, y_base, "b--", lw=2, label="baseline")
            plt.axvline(mu, color="purple", ls=":", lw=1.5, label=f"λmax={mu:.1f} nm")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Absorbance (au)")
            plt.title(f"UV-Vis fit ({peak_model} + poly deg {baseline_degree})")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return result


class UI :
    
    def __init__(self) :

        dt = dataTools()
        at = analysisTools()
        
        self.BufferNames = ['None']
        self.Names = ['']

        self.FoldersLabel = '-------Folders-------'
        self.FilesLabel = '-------Files-------'
        self.settingsFile = settingsFile
        
        
        try:
            with open(self.settingsFile, 'r') as stream:
                settings = yaml.safe_load(stream) or {}
        except FileNotFoundError:
            settings = {'folders': {'data': str(Path.cwd())}, 'files': {}}


        if os.path.isdir(settings['folders']['data']) :
            self.cwd = settings['folders']['data']
        else :
            self.cwd = str(Path(os.getcwd()))

        out = ipw.Output()
        anout = ipw.Output()
        
        uploadFile = ipw.FileUpload(
            accept='.csv, .xlsx',  # allowed file types
            multiple=False,
            description='Select File',
            layout=ipw.Layout(width='170px')
        )

        def on_uploadFile_change(change):
            with out :
                clear_output()
                if not uploadFile.value:
                    print("⚠️ No file uploaded.")
                    return
                data = dt.loadData(uploadFile.value)
                self.fig = dt.plot(data,Title=uploadFile.value[0]['name'])
                LowLim.max = int(data["Wavelength (nm)"].max())
                LowLim.min = int(data["Wavelength (nm)"].min())
                LowLim.value = 400
                UpLim.max = int(data["Wavelength (nm)"].max())
                UpLim.min = int(data["Wavelength (nm)"].min())
                UpLim.value = 700
                display(ipw.Box([LowLim, UpLim, Fit]))
            with anout :    
                clear_output
            self.data = data
        uploadFile.observe(on_uploadFile_change, names='value')

        def Fit_Clicked(b):
            with anout :
                clear_output()
                data = self.data
                res = at.fit_uvvis_lambdamax(data, fit_range=(LowLim.value, UpLim.value), baseline_degree=2, peak_model="gaussian", plot=True)
                # string = "$\\lambda_{\\max}$ = " + f"{res['lambda_max_nm']:.1f}" + " nm +/- " + f"{res['lambda_max_err_nm']:.1f}" +" nm"
                # display(md(string))
        Fit = ipw.Button(description="Fit")
        Fit.on_click(Fit_Clicked)
        
        LowLim = ipw.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description='Lower Limit:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
            )

        UpLim = ipw.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description='Upper Limit:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
            )

        display(ipw.Box([uploadFile]))

        display(out)
        display(anout)

    def get_folder_contents(self,folder):

        'Gets contents of folder, sorting by folder then files, hiding hidden things'
        folder = Path(folder)
        folders = [item.name for item in folder.iterdir() if item.is_dir() and not item.name.startswith('.')]
        files = [item.name for item in folder.iterdir() if item.is_file() and not item.name.startswith('.')]
        return sorted(folders), sorted(files)