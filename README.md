# Causal Discovery for Time Series with constraint-based model and PMIME measure

This repository contains codes for PC-PMIME, an algorithm to do causal discovery in nonlinear, stationary multivariate time series.
This is version 1, that still needs to get improved: In this code, orientation of edges is done only with asymetry, not with the PC rules. That can lead to spurious causality.

You may require the Tigramite package to run the test file, check : https://github.com/jakobrunge/tigramite

Additionally, to run the DYNOTEARS algorithm, i recommend you to upload the causalnex file : https://github.com/quantumblacklabs/causalnex/tree/develop/causalnex

Lastly, data and more details about the simulation process are available on https://github.com/ckassaad/causal_discovery_for_time_series or on https://dataverse.harvard.edu/dataverse/basic_causal_structures_additive_noise
