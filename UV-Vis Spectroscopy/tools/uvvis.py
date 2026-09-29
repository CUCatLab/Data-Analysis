import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import ipywidgets as ipw
from IPython.display import clear_output, display


# ============================================================
# Data tools
# ============================================================

class dataTools:

    def __init__(self):
        pass

    def loadData(self, uploaded_file):
        """
        Load one spectrum from an ipywidgets FileUpload object.

        The first two numeric columns are interpreted as:
            column 1: wavelength
            column 2: absorbance

        Supported formats:
            .csv
            .txt
            .dat
            .xlsx
            .xls

        Returns
        -------
        filename : str
        data : pandas.DataFrame
            Columns:
                Wavelength (nm)
                Absorbance (au)
        """

        filename = uploaded_file["name"]
        content = bytes(uploaded_file["content"])
        extension = Path(filename).suffix.lower()

        try:

            # ------------------------------------------------
            # Excel files
            # ------------------------------------------------

            if extension in [".xlsx", ".xls"]:

                data = pd.read_excel(
                    io.BytesIO(content)
                )

                numeric_columns = []

                for column in data.columns:

                    converted = pd.to_numeric(
                        data[column],
                        errors="coerce"
                    )

                    if converted.notna().sum() >= 10:
                        numeric_columns.append(column)

                if len(numeric_columns) < 2:
                    raise ValueError(
                        "The Excel file must contain at least "
                        "two numeric columns."
                    )

                data = data[
                    numeric_columns[:2]
                ].copy()

                data.columns = [
                    "Wavelength (nm)",
                    "Absorbance (au)"
                ]

                data["Wavelength (nm)"] = pd.to_numeric(
                    data["Wavelength (nm)"],
                    errors="coerce"
                )

                data["Absorbance (au)"] = pd.to_numeric(
                    data["Absorbance (au)"],
                    errors="coerce"
                )

            # ------------------------------------------------
            # Text-based files
            # ------------------------------------------------

            else:

                text = content.decode(
                    errors="ignore"
                )

                rows = []

                for line in text.splitlines():

                    # Try common delimiters
                    if "," in line:
                        parts = line.split(",")

                    elif "\t" in line:
                        parts = line.split("\t")

                    elif ";" in line:
                        parts = line.split(";")

                    else:
                        parts = line.split()

                    parts = [
                        part.strip()
                        for part in parts
                    ]

                    if len(parts) < 2:
                        continue

                    try:
                        wavelength = float(parts[0])
                        absorbance = float(parts[1])

                        rows.append(
                            (
                                wavelength,
                                absorbance
                            )
                        )

                    except ValueError:
                        # This skips headers and other text
                        continue

                if len(rows) < 10:
                    raise ValueError(
                        "No valid numeric UV-Vis data were found."
                    )

                data = pd.DataFrame(
                    rows,
                    columns=[
                        "Wavelength (nm)",
                        "Absorbance (au)"
                    ]
                )

            # ------------------------------------------------
            # Clean and sort
            # ------------------------------------------------

            data = (
                data
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .dropna()
                .sort_values("Wavelength (nm)")
                .drop_duplicates(
                    subset="Wavelength (nm)"
                )
                .reset_index(drop=True)
            )

            if len(data) < 10:
                raise ValueError(
                    "Fewer than 10 valid data points were found."
                )

            return filename, data

        except Exception as error:

            raise ValueError(
                f"Error reading {filename}: {error}"
            ) from error


# ============================================================
# Analysis tools
# ============================================================

class analysisTools:

    def __init__(self):
        pass

    def gaussian(
        self,
        x,
        amp,
        mu,
        sigma
    ):

        return amp * np.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )

    def lorentzian(
        self,
        x,
        amp,
        mu,
        gamma
    ):

        return amp * (
            gamma**2 /
            ((x - mu) ** 2 + gamma**2)
        )

    def poly_baseline(
        self,
        x,
        *coefficients
    ):
        """
        coefficients = b0, b1, ..., bN

        baseline =
            b0 + b1*x + b2*x**2 + ...
        """

        baseline = np.zeros_like(
            x,
            dtype=float
        )

        for degree, coefficient in enumerate(
            coefficients
        ):

            baseline += (
                coefficient * x**degree
            )

        return baseline

    def make_model(
        self,
        peak_model="gaussian",
        baseline_degree=2
    ):

        baseline_degree = int(
            baseline_degree
        )

        if baseline_degree < 0:
            raise ValueError(
                "Baseline degree must be at least zero."
            )

        if peak_model.lower() == "gaussian":

            def model(
                x,
                amp,
                mu,
                width,
                *baseline_coefficients
            ):

                return (
                    self.gaussian(
                        x,
                        amp,
                        mu,
                        width
                    )
                    +
                    self.poly_baseline(
                        x,
                        *baseline_coefficients
                    )
                )

            width_name = "sigma"

            def fwhm_from_width(width):
                return 2.35482 * width

        elif peak_model.lower() == "lorentzian":

            def model(
                x,
                amp,
                mu,
                width,
                *baseline_coefficients
            ):

                return (
                    self.lorentzian(
                        x,
                        amp,
                        mu,
                        width
                    )
                    +
                    self.poly_baseline(
                        x,
                        *baseline_coefficients
                    )
                )

            width_name = "gamma"

            def fwhm_from_width(width):
                return 2.0 * width

        else:

            raise ValueError(
                "Peak model must be Gaussian or Lorentzian."
            )

        number_baseline_parameters = (
            baseline_degree + 1
        )

        return (
            model,
            number_baseline_parameters,
            width_name,
            fwhm_from_width
        )

    def fit_uvvis_lambdamax(
        self,
        data,
        xcol="Wavelength (nm)",
        ycol="Absorbance (au)",
        fit_range=(400, 700),
        baseline_degree=2,
        peak_model="gaussian"
    ):
        """
        Fit one UV-Vis spectrum.

        Returns the calculated parameters and the arrays needed
        to draw the fit on an existing Matplotlib plot.
        """

        clean_data = (
            data[[xcol, ycol]]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        x = clean_data[
            xcol
        ].to_numpy(dtype=float)

        y = clean_data[
            ycol
        ].to_numpy(dtype=float)

        sort_index = np.argsort(x)

        x = x[sort_index]
        y = y[sort_index]

        lower_limit, upper_limit = fit_range

        fit_mask = (
            (x >= lower_limit)
            &
            (x <= upper_limit)
        )

        x = x[fit_mask]
        y = y[fit_mask]

        if x.size < 10:
            raise ValueError(
                "Not enough data points are present within "
                "the selected fitting range."
            )

        if np.ptp(x) <= 0:
            raise ValueError(
                "The wavelength range must be greater than zero."
            )

        (
            model,
            number_baseline_parameters,
            width_name,
            fwhm_from_width
        ) = self.make_model(
            peak_model=peak_model,
            baseline_degree=baseline_degree
        )

        # ----------------------------------------------------
        # Initial baseline estimate from the edges
        # ----------------------------------------------------

        number_points = x.size

        edge_size = max(
            5,
            int(0.08 * number_points)
        )

        edge_size = min(
            edge_size,
            max(
                2,
                number_points // 3
            )
        )

        edge_indices = np.r_[
            0:edge_size,
            number_points - edge_size:number_points
        ]

        x_edges = x[edge_indices]
        y_edges = y[edge_indices]

        polynomial_coefficients = np.polyfit(
            x_edges,
            y_edges,
            deg=baseline_degree
        )

        # np.polyfit returns highest power first.
        # Reverse to obtain b0, b1, ..., bN.
        baseline_initial = (
            polynomial_coefficients[::-1]
        )

        estimated_baseline = self.poly_baseline(
            x,
            *baseline_initial
        )

        baseline_subtracted = (
            y - estimated_baseline
        )

        # ----------------------------------------------------
        # Initial peak parameters
        # ----------------------------------------------------

        peak_index = np.argmax(
            baseline_subtracted
        )

        mu_initial = x[peak_index]

        amplitude_initial = max(
            1e-6,
            float(
                baseline_subtracted[peak_index]
            )
        )

        half_height = (
            0.5 * amplitude_initial
        )

        above_half_height = (
            baseline_subtracted >= half_height
        )

        if np.count_nonzero(
            above_half_height
        ) > 1:

            x_above_half = x[
                above_half_height
            ]

            fwhm_initial = (
                x_above_half.max()
                -
                x_above_half.min()
            )

        else:

            fwhm_initial = (
                0.10 * np.ptp(x)
            )

        if not np.isfinite(
            fwhm_initial
        ) or fwhm_initial <= 0:

            fwhm_initial = max(
                1.0,
                0.10 * np.ptp(x)
            )

        if peak_model.lower() == "gaussian":

            width_initial = max(
                1e-6,
                fwhm_initial / 2.35482
            )

        else:

            width_initial = max(
                1e-6,
                fwhm_initial / 2.0
            )

        # ----------------------------------------------------
        # Initial parameter array and bounds
        # ----------------------------------------------------

        initial_parameters = np.r_[
            [
                amplitude_initial,
                mu_initial,
                width_initial
            ],
            baseline_initial
        ]

        lower_bounds = [
            0.0,
            x.min(),
            1e-6
        ]

        upper_bounds = [
            np.inf,
            x.max(),
            np.ptp(x)
        ]

        for _ in range(
            number_baseline_parameters
        ):

            lower_bounds.append(-np.inf)
            upper_bounds.append(np.inf)

        lower_bounds = np.array(
            lower_bounds,
            dtype=float
        )

        upper_bounds = np.array(
            upper_bounds,
            dtype=float
        )

        # ----------------------------------------------------
        # Fit
        # ----------------------------------------------------

        fitted_parameters, covariance = curve_fit(
            model,
            x,
            y,
            p0=initial_parameters,
            bounds=(
                lower_bounds,
                upper_bounds
            ),
            maxfev=50000
        )

        amplitude = fitted_parameters[0]
        lambda_max = fitted_parameters[1]
        width = fitted_parameters[2]

        baseline_parameters = (
            fitted_parameters[3:]
        )

        y_fit = model(
            x,
            *fitted_parameters
        )

        y_baseline = self.poly_baseline(
            x,
            *baseline_parameters
        )

        y_peak = (
            y_fit - y_baseline
        )

        # ----------------------------------------------------
        # Fit statistics
        # ----------------------------------------------------

        residuals = y - y_fit

        ss_residual = np.sum(
            residuals**2
        )

        ss_total = np.sum(
            (y - np.mean(y))**2
        )

        if ss_total > 0:

            r_squared = (
                1
                -
                ss_residual / ss_total
            )

        else:

            r_squared = np.nan

        if (
            covariance is not None
            and
            np.all(
                np.isfinite(covariance)
            )
        ):

            parameter_errors = np.sqrt(
                np.diag(covariance)
            )

        else:

            parameter_errors = np.full(
                len(fitted_parameters),
                np.nan
            )

        lambda_max_error = (
            parameter_errors[1]
        )

        return {
            "lambda_max_nm": float(
                lambda_max
            ),
            "lambda_max_err_nm": float(
                lambda_max_error
            ),
            "amp": float(amplitude),
            width_name: float(width),
            "fwhm_nm": float(
                fwhm_from_width(width)
            ),
            "baseline_coeffs_b0_to_bN": [
                float(value)
                for value in baseline_parameters
            ],
            "r2": float(r_squared),
            "popt": fitted_parameters,
            "pcov": covariance,
            "x_fit": x,
            "y_data": y,
            "y_fit": y_fit,
            "y_baseline": y_baseline,
            "y_peak": y_peak
        }


# ============================================================
# User interface
# ============================================================

class UI:

    def __init__(self):

        self.dt = dataTools()
        self.at = analysisTools()

        # All loaded spectra are stored here:
        #
        # self.spectra = {
        #     "sample1.csv": DataFrame,
        #     "sample2.csv": DataFrame
        # }

        self.spectra = {}

        # Fit results for the currently selected spectra
        self.fit_results = {}

        # If True, the current plot includes fit results
        self.showing_fits = False

        # ----------------------------------------------------
        # File upload
        # ----------------------------------------------------

        self.uploadFile = ipw.FileUpload(
            accept=".csv,.txt,.dat,.xlsx,.xls",
            multiple=False,
            description="Add Spectrum",
            layout=ipw.Layout(
                width="170px"
            )
        )

        # ----------------------------------------------------
        # Spectrum selection list
        # ----------------------------------------------------

        self.spectrumList = ipw.SelectMultiple(
            options=[],
            value=(),
            description="Spectra:",
            rows=9,
            layout=ipw.Layout(
                width="500px",
                height="210px"
            ),
            style={
                "description_width": "70px"
            }
        )

        # ----------------------------------------------------
        # Fit controls
        # ----------------------------------------------------

        self.LowLim = ipw.IntSlider(
            value=400,
            min=200,
            max=800,
            step=1,
            description="Lower Limit:",
            disabled=False,
            continuous_update=False,
            readout=True,
            readout_format="d",
            layout=ipw.Layout(
                width="450px"
            ),
            style={
                "description_width": "100px"
            }
        )

        self.UpLim = ipw.IntSlider(
            value=700,
            min=200,
            max=800,
            step=1,
            description="Upper Limit:",
            disabled=False,
            continuous_update=False,
            readout=True,
            readout_format="d",
            layout=ipw.Layout(
                width="450px"
            ),
            style={
                "description_width": "100px"
            }
        )

        self.peakModel = ipw.Dropdown(
            options=[
                ("Gaussian", "gaussian"),
                ("Lorentzian", "lorentzian")
            ],
            value="gaussian",
            description="Peak model:",
            layout=ipw.Layout(
                width="260px"
            ),
            style={
                "description_width": "90px"
            }
        )

        # ipywidgets does not have IntDropdown.
        # A regular Dropdown can contain integer values.
        self.baselineDegree = ipw.Dropdown(
            options=[
                ("Constant, degree 0", 0),
                ("Linear, degree 1", 1),
                ("Quadratic, degree 2", 2),
                ("Cubic, degree 3", 3)
            ],
            value=2,
            description="Baseline:",
            layout=ipw.Layout(
                width="260px"
            ),
            style={
                "description_width": "75px"
            }
        )

        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        self.fitButton = ipw.Button(
            description="Fit Selected",
            button_style="primary",
            icon="check",
            tooltip=(
                "Fit all selected spectra and display "
                "the fits and lambda max values"
            ),
            layout=ipw.Layout(
                width="150px"
            )
        )

        self.clearFitButton = ipw.Button(
            description="Clear Fits",
            button_style="",
            icon="eraser",
            tooltip=(
                "Remove the fitted curves and return "
                "to the raw spectra"
            ),
            layout=ipw.Layout(
                width="130px"
            )
        )

        self.removeButton = ipw.Button(
            description="Remove Selected",
            button_style="warning",
            icon="trash",
            layout=ipw.Layout(
                width="170px"
            )
        )

        self.clearButton = ipw.Button(
            description="Clear All",
            button_style="danger",
            icon="times",
            layout=ipw.Layout(
                width="120px"
            )
        )

        # ----------------------------------------------------
        # Output widgets
        # ----------------------------------------------------

        self.statusOutput = ipw.Output()
        self.plotOutput = ipw.Output()
        self.resultsOutput = ipw.Output()

        # ----------------------------------------------------
        # Widget callbacks
        # ----------------------------------------------------

        self.uploadFile.observe(
            self.on_uploadFile_change,
            names="value"
        )

        self.spectrumList.observe(
            self.on_selection_change,
            names="value"
        )

        self.LowLim.observe(
            self.on_fit_setting_change,
            names="value"
        )

        self.UpLim.observe(
            self.on_fit_setting_change,
            names="value"
        )

        self.peakModel.observe(
            self.on_fit_setting_change,
            names="value"
        )

        self.baselineDegree.observe(
            self.on_fit_setting_change,
            names="value"
        )

        self.fitButton.on_click(
            self.fit_selected
        )

        self.clearFitButton.on_click(
            self.clear_fits
        )

        self.removeButton.on_click(
            self.remove_selected
        )

        self.clearButton.on_click(
            self.clear_all
        )

        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        upload_controls = ipw.HBox(
            [
                self.uploadFile,
                self.removeButton,
                self.clearButton
            ],
            layout=ipw.Layout(
                gap="10px",
                align_items="center",
                flex_flow="row wrap"
            )
        )

        fit_options = ipw.HBox(
            [
                self.peakModel,
                self.baselineDegree
            ],
            layout=ipw.Layout(
                gap="10px",
                flex_flow="row wrap"
            )
        )

        fit_buttons = ipw.HBox(
            [
                self.fitButton,
                self.clearFitButton
            ],
            layout=ipw.Layout(
                gap="10px"
            )
        )

        fit_controls = ipw.VBox(
            [
                self.LowLim,
                self.UpLim,
                fit_options,
                fit_buttons
            ],
            layout=ipw.Layout(
                gap="8px"
            )
        )

        main_controls = ipw.HBox(
            [
                self.spectrumList,
                fit_controls
            ],
            layout=ipw.Layout(
                gap="20px",
                align_items="flex-start",
                flex_flow="row wrap"
            )
        )

        display(
            ipw.VBox(
                [
                    ipw.HTML(
                        "<h3>UV-Vis Spectra and "
                        "λ<sub>max</sub> Fitting</h3>"
                    ),
                    ipw.HTML(
                        "<p>"
                        "Add spectra one at a time, select one or "
                        "more spectra from the list, and then click"
                        "<b>Fit Selected</b>."
                        "</p>"
                    ),
                    upload_controls,
                    self.statusOutput,
                    main_controls,
                    self.plotOutput,
                    self.resultsOutput
                ],
                layout=ipw.Layout(
                    gap="8px"
                )
            )
        )

    # ========================================================
    # Utility methods
    # ========================================================

    def _get_uploaded_files(self):
        """
        Normalize FileUpload output for different ipywidgets
        versions.

        ipywidgets 7 may return a dictionary.
        ipywidgets 8 normally returns a tuple.
        """

        value = self.uploadFile.value

        if isinstance(value, dict):

            uploaded_files = []

            for filename, file_info in value.items():

                normalized_file = dict(
                    file_info
                )

                normalized_file.setdefault(
                    "name",
                    filename
                )

                uploaded_files.append(
                    normalized_file
                )

            return uploaded_files

        return list(value)

    def _selected_names(self):

        return list(
            self.spectrumList.value
        )

    def _make_unique_name(
        self,
        filename
    ):
        """
        Prevent a new file from silently replacing a spectrum
        that has the same filename.
        """

        if filename not in self.spectra:
            return filename

        path = Path(filename)

        stem = path.stem
        suffix = path.suffix

        counter = 2

        while True:

            new_name = (
                f"{stem} ({counter}){suffix}"
            )

            if new_name not in self.spectra:
                return new_name

            counter += 1

    def _update_spectrum_list(
        self,
        selected_names=None
    ):

        self.spectrumList.options = list(
            self.spectra.keys()
        )

        if selected_names is None:
            return

        valid_names = [
            name
            for name in selected_names
            if name in self.spectra
        ]

        self.spectrumList.value = tuple(
            valid_names
        )

    def _update_slider_limits(self):
        """
        Update the wavelength slider limits using all loaded
        spectra.
        """

        if not self.spectra:
            return

        minimum_wavelength = min(
            data["Wavelength (nm)"].min()
            for data in self.spectra.values()
        )

        maximum_wavelength = max(
            data["Wavelength (nm)"].max()
            for data in self.spectra.values()
        )

        minimum_wavelength = int(
            np.floor(minimum_wavelength)
        )

        maximum_wavelength = int(
            np.ceil(maximum_wavelength)
        )

        if minimum_wavelength >= maximum_wavelength:
            maximum_wavelength = (
                minimum_wavelength + 1
            )

        # Temporarily stop slider observation so changing the
        # bounds does not repeatedly redraw the plot.
        self.LowLim.unobserve(
            self.on_fit_setting_change,
            names="value"
        )

        self.UpLim.unobserve(
            self.on_fit_setting_change,
            names="value"
        )

        try:

            self.LowLim.min = minimum_wavelength
            self.LowLim.max = maximum_wavelength

            self.UpLim.min = minimum_wavelength
            self.UpLim.max = maximum_wavelength

            lower_default = max(
                minimum_wavelength,
                400
            )

            upper_default = min(
                maximum_wavelength,
                700
            )

            if lower_default >= upper_default:

                lower_default = (
                    minimum_wavelength
                )

                upper_default = (
                    maximum_wavelength
                )

            self.LowLim.value = int(
                lower_default
            )

            self.UpLim.value = int(
                upper_default
            )

        finally:

            self.LowLim.observe(
                self.on_fit_setting_change,
                names="value"
            )

            self.UpLim.observe(
                self.on_fit_setting_change,
                names="value"
            )

    # ========================================================
    # Plotting
    # ========================================================

    def draw_plot(
        self,
        include_fits=False
    ):
        """
        Draw the selected spectra.

        If include_fits is True, fitted curves and lambda-max
        labels are added to this same figure.
        """

        selected_names = (
            self._selected_names()
        )

        with self.plotOutput:

            clear_output(wait=True)

            if not selected_names:

                print(
                    "Select one or more spectra from the list."
                )

                return

            fig, ax = plt.subplots(
                figsize=(11, 7)
            )

            colors = plt.rcParams[
                "axes.prop_cycle"
            ].by_key()["color"]

            for spectrum_number, name in enumerate(
                selected_names
            ):

                data = self.spectra[name]

                color = colors[
                    spectrum_number % len(colors)
                ]

                # --------------------------------------------
                # Raw spectrum
                # --------------------------------------------

                ax.plot(
                    data["Wavelength (nm)"],
                    data["Absorbance (au)"],
                    color=color,
                    lw=1.7,
                    alpha=0.85,
                    label=name
                )

                # --------------------------------------------
                # Fit, if available
                # --------------------------------------------

                if (
                    include_fits
                    and
                    name in self.fit_results
                ):

                    result = self.fit_results[name]

                    lambda_max = result[
                        "lambda_max_nm"
                    ]

                    lambda_error = result[
                        "lambda_max_err_nm"
                    ]

                    ax.plot(
                        result["x_fit"],
                        result["y_fit"],
                        color=color,
                        lw=2.4,
                        ls="--",
                        label=(
                            f"{name} fit: "
                            f"λmax = {lambda_max:.1f} nm"
                        )
                    )

                    ax.axvline(
                        lambda_max,
                        color=color,
                        lw=1.2,
                        ls=":",
                        alpha=0.9
                    )

                    peak_plot_index = np.argmin(
                        np.abs(
                            result["x_fit"]
                            -
                            lambda_max
                        )
                    )

                    y_at_lambda_max = result[
                        "y_fit"
                    ][peak_plot_index]

                    if np.isfinite(
                        lambda_error
                    ):

                        label_text = (
                            f"λmax = {lambda_max:.1f} "
                            f"± {lambda_error:.1f} nm"
                        )

                    else:

                        label_text = (
                            f"λmax = {lambda_max:.1f} nm"
                        )

                    ax.annotate(
                        label_text,
                        xy=(
                            lambda_max,
                            y_at_lambda_max
                        ),
                        xytext=(7, 8),
                        textcoords="offset points",
                        color=color,
                        fontsize=10,
                        fontweight="bold",
                        rotation=90,
                        ha="left",
                        va="bottom"
                    )

            # Show the selected fitting range
            ax.axvspan(
                self.LowLim.value,
                self.UpLim.value,
                color="gray",
                alpha=0.06,
                label="Fit range"
            )

            ax.set_xlabel(
                "Wavelength (nm)",
                fontsize=14
            )

            ax.set_ylabel(
                "Absorbance (au)",
                fontsize=14
            )

            if include_fits:

                ax.set_title(
                    "UV-Vis Spectra with Fitted λmax Values",
                    fontsize=14
                )

            else:

                ax.set_title(
                    "Selected UV-Vis Spectra",
                    fontsize=14
                )

            ax.tick_params(
                axis="both",
                which="both",
                labelsize=12,
                direction="in"
            )

            ax.minorticks_on()

            ax.legend(
                frameon=False,
                fontsize=9,
                bbox_to_anchor=(1.02, 1),
                loc="upper left"
            )

            fig.tight_layout()
            plt.show()

    # ========================================================
    # Widget callbacks
    # ========================================================

    def on_uploadFile_change(
        self,
        change
    ):

        uploaded_files = (
            self._get_uploaded_files()
        )

        if not uploaded_files:
            return

        newly_loaded = []

        with self.statusOutput:

            clear_output()

            for uploaded_file in uploaded_files:

                try:

                    filename, data = (
                        self.dt.loadData(
                            uploaded_file
                        )
                    )

                    unique_filename = (
                        self._make_unique_name(
                            filename
                        )
                    )

                    self.spectra[
                        unique_filename
                    ] = data

                    newly_loaded.append(
                        unique_filename
                    )

                    print(
                        f"Loaded: {unique_filename} "
                        f"({len(data)} data points)"
                    )

                except Exception as error:

                    print(
                        f"Could not load file: {error}"
                    )

        if newly_loaded:

            self._update_slider_limits()

            # Select the newly uploaded spectrum without
            # removing previously selected spectra.
            selected_names = list(
                self.spectrumList.value
            )

            selected_names.extend(
                newly_loaded
            )

            self._update_spectrum_list(
                selected_names=selected_names
            )

            self.fit_results.clear()
            self.showing_fits = False

            self.draw_plot(
                include_fits=False
            )

        # Reset the upload widget where supported.
        # This allows the same filename to be uploaded again.
        try:
            self.uploadFile.value = ()
        except Exception:
            pass

    def on_selection_change(
        self,
        change
    ):
        """
        Automatically plot the raw selected spectra whenever
        the selection changes.
        """

        self.fit_results.clear()
        self.showing_fits = False

        with self.resultsOutput:
            clear_output()

        if self._selected_names():

            self.draw_plot(
                include_fits=False
            )

        else:

            with self.plotOutput:
                clear_output()

    def on_fit_setting_change(
        self,
        change
    ):
        """
        When a fitting setting changes, clear stale fit results
        and return to the raw-spectrum plot.
        """

        self.fit_results.clear()
        self.showing_fits = False

        with self.resultsOutput:
            clear_output()

        if self._selected_names():

            self.draw_plot(
                include_fits=False
            )

    def fit_selected(
        self,
        button=None
    ):
        """
        Fit every selected spectrum and redraw the existing plot
        with the raw data, fitted curves, and lambda-max labels.
        """

        selected_names = (
            self._selected_names()
        )

        with self.resultsOutput:

            clear_output(wait=True)

            if not selected_names:

                print(
                    "Select one or more spectra before fitting."
                )

                return

            if (
                self.LowLim.value
                >=
                self.UpLim.value
            ):

                print(
                    "The lower fitting limit must be below "
                    "the upper fitting limit."
                )

                return

            fit_range = (
                self.LowLim.value,
                self.UpLim.value
            )

            self.fit_results = {}

            results_for_table = []
            failed_fits = []

            for name in selected_names:

                data = self.spectra[name]

                try:

                    result = (
                        self.at.fit_uvvis_lambdamax(
                            data=data,
                            fit_range=fit_range,
                            baseline_degree=(
                                self.baselineDegree.value
                            ),
                            peak_model=(
                                self.peakModel.value
                            )
                        )
                    )

                    self.fit_results[
                        name
                    ] = result

                    results_for_table.append(
                        {
                            "Spectrum": name,
                            "Lambda max (nm)": (
                                result[
                                    "lambda_max_nm"
                                ]
                            ),
                            "Uncertainty (nm)": (
                                result[
                                    "lambda_max_err_nm"
                                ]
                            ),
                            "FWHM (nm)": (
                                result[
                                    "fwhm_nm"
                                ]
                            ),
                            "R squared": (
                                result[
                                    "r2"
                                ]
                            )
                        }
                    )

                except Exception as error:

                    failed_fits.append(
                        f"{name}: {error}"
                    )

            if self.fit_results:

                self.showing_fits = True

                # Redraw the same plot output with the fit data
                self.draw_plot(
                    include_fits=True
                )

            if results_for_table:

                results_table = pd.DataFrame(
                    results_for_table
                )

                results_table[
                    "Lambda max (nm)"
                ] = results_table[
                    "Lambda max (nm)"
                ].round(2)

                results_table[
                    "Uncertainty (nm)"
                ] = results_table[
                    "Uncertainty (nm)"
                ].round(2)

                results_table[
                    "FWHM (nm)"
                ] = results_table[
                    "FWHM (nm)"
                ].round(2)

                results_table[
                    "R squared"
                ] = results_table[
                    "R squared"
                ].round(5)

                display(
                    results_table
                )

            if failed_fits:

                print(
                    "\nFits that could not be completed:"
                )

                for failure in failed_fits:

                    print(
                        f"  {failure}"
                    )

    def clear_fits(
        self,
        button=None
    ):

        self.fit_results.clear()
        self.showing_fits = False

        with self.resultsOutput:
            clear_output()

        if self._selected_names():

            self.draw_plot(
                include_fits=False
            )

    def remove_selected(
        self,
        button=None
    ):

        selected_names = (
            self._selected_names()
        )

        if not selected_names:
            return

        for name in selected_names:

            self.spectra.pop(
                name,
                None
            )

            self.fit_results.pop(
                name,
                None
            )

        remaining_names = list(
            self.spectra.keys()
        )

        self.spectrumList.options = (
            remaining_names
        )

        self.spectrumList.value = ()

        self.showing_fits = False

        with self.statusOutput:

            clear_output()

            print(
                f"Removed {len(selected_names)} "
                f"spectrum/spectra."
            )

        with self.plotOutput:
            clear_output()

        with self.resultsOutput:
            clear_output()

        if self.spectra:

            self._update_slider_limits()

    def clear_all(
        self,
        button=None
    ):

        self.spectra.clear()
        self.fit_results.clear()

        self.spectrumList.options = []
        self.spectrumList.value = ()

        self.showing_fits = False

        with self.statusOutput:

            clear_output()

            print(
                "All spectra were removed."
            )

        with self.plotOutput:
            clear_output()

        with self.resultsOutput:
            clear_output()

    def get_folder_contents(
        self,
        folder
    ):

        folder = Path(folder)

        folders = [
            item.name
            for item in folder.iterdir()
            if item.is_dir()
            and not item.name.startswith(".")
        ]

        files = [
            item.name
            for item in folder.iterdir()
            if item.is_file()
            and not item.name.startswith(".")
        ]

        return (
            sorted(folders),
            sorted(files)
        )