---
title: "SNAzzy: an image processing pipeline for investigating global Synchronous Network Activity"
tags:
  - Calcium imaging
  - Widefield microscopy
  - Image analysis
  - Drosophila
  - Spontaneous Neural Activity
  - Neurodevelopment
  - Circuit Wiring
authors:
  - name: Carlos Damiani Paiva
    orcid: 0009-0007-6658-2620
    affiliation: "1, 2"
  - name: Alana J. Evora
    orcid: 0009-0002-3174-7839
    affiliation: "1, 2"
  - name: Shirui Zheng
    orcid: 0009-0006-1836-7208
    affiliation: "1, 2"
  - name: Arnaldo Carreira-Rosario
    orcid: 0000-0003-0202-3858
    corresponding: true
    affiliation: "1, 2"
affiliations:
  - name: Department of Biology, Duke University, Durham, NC 27708
    index: 1
  - name: Department of Neurobiology, Duke University, Durham, NC 27708
    index: 2
date: 8 October 2025
bibliography: paper.bib
---

# Summary

Genetically encoded fluorescent indicators are powerful tools for monitoring biological processes in live samples [@lin:2016; @nakai:2001].
When combined with a large field of view, a single time-lapse recording has the potential to capture many specimens, facilitating high-throughput data collection.
However, the simultaneous recording of many biological samples across time points produces large, multidimensional datasets that are challenging to process and analyze.
We present `SNAzzy`, a Python package for studying synchronous network activity (SNA) in Drosophila embryos via high-throughput microscopy.
SNA is a hallmark of developing nervous systems [@wu:2024; @blankenship:2009; @akin:2020], often studied using genetically encoded calcium indicators to monitor neural activity in vivo.
`SNAzzy` processes and analyzes time-lapse datasets taken from live samples using fluorescent widefield microscopy.
Each dataset contains dozens of individual specimens in the same field of view and thousands of time points.
The software offers individual specimen cropping for optimization of storage and processing, adaptive regions of interest for quantification of fluorescence and changes in morphology over time, a custom peak detection algorithm, and a graphical user interface for data visualization, curation, and dataset comparison.
This tool can be readily applied to analyze fluorescent intensities in time-lapse microscopy experiments that involve simultaneous imaging of multiple samples, particularly small-sized specimens [@donoughe:2018; @avasthi:2023]. 

# Statement of need

During synchronous network activity (SNA), many neurons fire synchronously, generating waves of activity that span across large portions of the nervous system [@blankenship:2009; @wu:2024; @akin:2020].
In Drosophila embryos, SNA typically lasts 4 hours, during which the nervous system undergoes a stereotyped morphological change via ventral nerve cord condensation [@crisp:2008; @carreira:2021].
To gain an understanding of SNA, it is essential to quantify waves of activity in the nervous system while also tracking morphology as a proxy of neurodevelopment.
For these reasons, we combine a commonly used genetically encoded calcium indicator (GECI) that reports neural activity with a structural fluorophore [@carreira:2021].
The structural fluorophore signal remains stable, independent of neural activity, making it suitable for continuous tracking morphology of the nerve cord.
To record many embryos during SNA, we use a wide-field fluorescence microscopy system that captures the GECI and structural fluorophore signal of dozens of developing embryos for over 5 hours.

We were unable to find a tool designed for widefield microscopy that rapidly processes multiple specimens, quantifies levels of fluorophore activity, and incorporates a peak-finding algorithm suitable for global calcium traces.
`SNAzzy` is designed to investigate global levels of neural activity across multiple developing embryos simultaneously.

## Tracking of multiple “adaptive  ROIs”  

To the best of our knowledge, there are no other packages that provide functionality for automated parsing of raw images of many live specimens into activity and morphological quantifications.
Other studies have employed manual selection of regions of interest (ROIs) and used static ROIs [@akin:2020; @menzies:2024; @ardiel:2022; @carreira:2021].
Manual selection often generates imprecise ROIs, which can lead to inaccurate quantifications, and is also cumbersome and prone to human error.
Static ROIs are not reliable for detecting the fluorescent signal of live specimens that change in morphology and move while imaging.
`SNAzzy` fills these gaps as an accessible pipeline for the automated analysis of multiple live samples in parallel.
The pipeline generates an “adaptive ROI” that changes frame-by-frame for each specimen.
This enables the accurate tracking of fluorescence intensity as well as changes in tissue morphology or size.
`SNAzzy`’s design provides an automated, modular, and fully auditable workflow, and ultimately contributes to more reproducible and comparable results across experiments.

## Capturing global Calcium dynamics

To the best of our knowledge, there are no open-source packages that provide tools for performing automated data analysis and quantification of global calcium dynamics.
Most open-source tools available for analyzing neural activity using GECI focus on segmenting individual neurons within a single specimen.
`CaImAn` [@giovannucci:2019], and `Suite2p` [@pachitariu:2016] are among the most widely used.
These packages detect calcium dynamics and use individual neuron statistics to perform spike inference, but do not offer direct peak detection on the calcium signal.
Furthermore, they are optimized for two-photon microscopy as opposed to wide-field microscopy.
`SNAzzy` provides a series of automated analyses and quantifications to analyze global calcium levels in time-series acquired with widefield microscopes.

![Schematic of the SNAzzy pipeline.
Time-lapse taken from fluorescent widefield microscopes (raw data) enters the processing stage (green).
The processing stage outputs two types of CSV files: time series of signal intensities from each recorded channel and ROI length.
CSV files enter the analysis stage (blue) to generate normalized fluorescent traces and detect peaks along with other signal processing metrics.
These initial traces can be visualized to curate the data.
Curation generates a configuration file that works as metadata across platforms and users.
Curated data can be reanalyzed and used to visualize final data and compare across groups (yellow).
Analysis and output stages are performed in the GUI (red dashed box), along with other metrics.
Dashed arrows indicate optional steps.\label{fig:fig1}](figures/snazzy-fig1.png)

# Pipeline Description

The initial input for `SNAzzy` \autoref{fig:fig1} is raw time-lapse imaging data containing multiple embryos.
Each embryo expresses a GECI (dynamic fluorophore) and a structural fluorophore.
Fluorophores are imaged in different optical channels.

The first pipeline step converts the raw data to TIF format, thereby avoiding compatibility issues that may arise when parsing different proprietary formats \autoref{fig:fig1}.
All embryos are then segmented using histogram equalization, followed by intensity threshold binarization [@otsu:1979].
Boxes surrounding the segments are cropped into individual time-lapses for each embryo.
Cropping results in a substantial memory reduction, as most background pixels are removed, with cropped images typically accounting for around 40% of the original size.

The next step is to process each individual specimen.
First, the ROI, which in our case is the entire central nervous system (CNS), is defined by binarizing the structural channel and selecting the largest connected component.
This process is repeated at every time point to generate an “adaptive ROI”.
From these adaptive ROI, the average signal intensity for both channels is extracted.
The results are saved as CSV files and are the basis for downstream analysis.

![ROI length measurement algorithm and validation.
A) Steps to calculate the ROI length.
The ROI length is calculated by estimating the centerline (red line) using points of maximum (dots) in the distance transform, followed by RANSAC to ignore outliers (orange dots).
B) Validation of the method as relative error (measured - annotated) / annotated.
Each whisker bar summarizes the relative error for frames taken at intervals of 50 timepoints.
C) Comparison of absolute values over a time series for three representative embryos.\label{fig:fig2}](figures/snazzy-fig2.png)

The ROI is also used to measure the length of the CNS \autoref{fig:fig2}.
Drosophila embryo CNS length serves as an internal proxy for neurodevelopmental stages, enabling more accurate comparisons across embryos [@carreira:2021].
The CNS length is calculated by centerline estimation.
First, a distance transform is applied to the binarized image, and local maxima points are detected.
Depending on the embryo's orientation, some points may be part of the brain lobes and must be filtered out to accurately measure the CNS length.
To obtain a robust centerline estimate that can ignore outliers, we use RANSAC [@fischler:1981] over the local maxima points and measure the overlap between the fitted line and the binary image.
CNS length is also detected frame by frame and exported as a CSV file \autoref{fig:fig1}.

![Peak detection algorithm.
A low-pass filter (orange line) is applied to the ∆F/F signal (black line) to remove fast transients.
The peak in the filtered signal (orange dot) is then ported back to the ∆F/F (blue dot) signal by selecting the leftmost peak within a search window (blue lines).\label{fig:fig3}](figures/snazzy-fig3.png)

The package utilizes average signal intensity measurements to calculate ∆F/F traces and peaks.
For ∆F/F, we first calculate the ratiometric signal (dynamic signal / structural signal) and then its baseline, which is defined as the average of the N lowest values within a sliding window.
The generated ∆F/F traces contain long-duration bouts of activity with superimposed fast transients \autoref{fig:fig3}.
The former represents the bursts of activity and is the most relevant for the initial analysis.
To mark only these more prolonged bouts, we apply a low-pass frequency filter to omit transients.
Peaks in the filtered trace are detected using SciPy [@virtanen:2020].
Finally, the detected peaks are ported to the original ∆F/F signal.

Results can be visualized and curated in a graphical user interface (GUI) implemented in `PyQt6` \autoref{fig:fig4}.
During curation, researchers can modify data analysis parameters, which are persisted in a JSON configuration file and utilized by the core analysis code across different machines and users.
Finally, a large number of different metrics and representations derived from ∆F/F, CNS length, and peaks can be visualized and plotted using the GUI.
These include SNA onset, burst duration and spectrograms, among others.

![GUI for data validation, curation, visualization and plotting.
Initial GUI screen.
A ∆F/F trace (white) and the corresponding peaks (magenta dots) are shown.
The low-passed signal (green line) is used as a reference to determine peaks.
The GUI enables the modification of analysis parameters, visualization of data, and comparison of metrics across groups of experiments, as well as manual adjustment of peak data.\label{fig:fig4}](figures/snazzy-fig4.png)

In conclusion, genetically encoded fluorescent indicators and microscopy systems are evolving rapidly, increasing the data acquisition throughput.
Custom open-source tools are needed to handle such large data files.
`SNAzzy` addresses this by offering an automated, scalable, and user-friendly platform for analyzing synchronous network activity in developing embryos.
As an open and versatile solution, `SNAzzy` offers tools for a broader range of applications in time-lapse fluorescence imaging across diverse biological systems.

# Acknowledgments

We acknowledge Newt PenkoffLidbeck and D. Berfin Azizoglu for feedback on the manuscript.
This work was partially funded by NINDS and the BRAIN initiative (R00NS119295).

# References
