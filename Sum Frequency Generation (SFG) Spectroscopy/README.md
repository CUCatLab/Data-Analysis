# Sum Frequency Generation (SFG) Spectroscopy Analysis

This repository provides tools for processing and analyzing Sum Frequency Generation (SFG) spectroscopy data. SFG is a nonlinear optical technique used to probe molecular structure and orientation at interfaces, with applications in surface chemistry, biology, and materials science.

## Overview

The program reads raw data and uses user-defined settings to perform fitting, baseline correction, and/or peak deconvolution. The main components include:

* **Raw Data Input** : Accepts standard data output files.
* **Analysis File** : Defines metadata and fitting parameters.
* **Output** : Generates processed plots, fit results, and summary tables.

## Analysis File

The analysis file is a required input that provides configuration details for processing the data. It can contain:

* **Metadata** : Sample identifiers, experimental conditions (e.g., heating rate, temperature range).
* **Fitting Parameters** : Initial guesses, constraints, and model settings for peak fitting.
* **Comments** : Optional notes for documenting the experiment.

See the included `Analysis File - Example.yaml` for formatting guidelines and supported fields. Accurate setup of the analysis file ensures consistent and reproducible results across datasets.
