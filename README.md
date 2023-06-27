# HandsfreeReco

This folder contains example code for applying a reconstruction parameter free MPI reconstruction based on the kaczmarz algorithm.
Given is an example on dilution series of a phantom of a realistic rat-kidney (example.jl).

The method as well as the data is described in the associated publication [...].


## Installation

In order to use this code one first has to download [Julia](https://julialang.org/) (version 1.7 or later), clone this repository and navigate to the folder in the command line. The example scripts automatically activate the environment and install all necessary packages.

## Execution
After installation the example code can be executed by running `julia` and entering
```julia
include("example.jl")
```
.
This will first download all data, perform a parameter free reconstruction and plot the reconstrucion results in a seperate window.

## Open MPI Data

The measurement data associated to this project is about 2.1 GB large and will be downloaded and stored automatically, when the code is executed for the first time.
It is published under a [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode) license and can be found here:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5006979.svg)](https://doi.org/10.5281/zenodo.5006979)
