# Temperature Programmed Desorption (TPD)

This program is for analyzing Temperature Programmed Desorption (TPD) data. TPD is a surface science technique used to study how molecules desorb from a material's surface as the temperature increases over time. The software helps extract key parameters such as desorption energies, peak areas, and reaction orders from experimental TPD spectra.

## Overview

The program reads raw data and uses user-defined settings to perform fitting, baseline correction, and/or peak deconvolution. The main components include:

* **Raw Data Input** : Accepts standard data files.
* **Analysis File** : Defines metadata and fitting parameters.
* **Output** : Generates processed plots, fit results, and summary tables.

## Analysis File

The analysis file is a required input that provides configuration details for processing the data. It can contain:

* **Metadata** : Sample identifiers, experimental conditions (e.g., heating rate, temperature range).
* **Fitting Parameters** : Initial guesses, constraints, and model settings for peak fitting.
* **Comments** : Optional notes for documenting the experiment.

See the included `Analysis File - Example.yaml` for formatting guidelines and supported fields. Accurate setup of the analysis file ensures consistent and reproducible results across datasets.
