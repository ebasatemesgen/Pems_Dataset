# PEMS Dataset: Traffic Flow Prediction with PEMS04 and STGODE Model

This repository contains the PEMS04 dataset and resources for traffic flow prediction using the STGODE model. The PEMS04 dataset is part of the PEMS (PeMS Traffic Monitoring) dataset collection, which contains real-time traffic data collected from loop detectors on California highways.

## Table of Contents

1. [Introduction](#introduction)
2. [PEMS04 Dataset](#pems04-dataset)
3. [STGODE Model](#stgode-model)
4. [Usage](#usage)
5. [Credits](#credits)

## Introduction

Traffic congestion is a growing issue in urban areas, resulting in economic loss and environmental problems. Accurate traffic flow predictions can help traffic management authorities implement effective traffic control and provide route suggestions to travelers. The PEMS04 dataset is a valuable resource for researchers and developers working on traffic flow prediction and related problems.

## PEMS04 Dataset

The PEMS04 dataset is a subset of the PEMS (PeMS Traffic Monitoring) dataset, which contains real-time traffic data collected from loop detectors on California highways. The dataset includes traffic speed, occupancy, and flow measurements.

PEMS04 is specifically focused on the California State Route 4 (SR 4) highway, which is part of the larger PEMS dataset.

## STGODE Model

The Spatio-Temporal Graph ODE Networks (STGODE) model is a deep learning model designed for traffic flow prediction. It leverages graph neural networks and ordinary differential equations to capture the spatio-temporal dependencies in the traffic data. This model has shown promising results in predicting traffic flow patterns with lower loss.

For more information and implementation details of the STGODE model, please visit the [STGODE repository](https://github.com/square-coder/STGODE).

## Usage

To use the PEMS04 dataset in your research or development projects, you can download it using the following command:

<pre>
git clone https://github.com/ebagirma/Pems_Dataset.git
</pre>

Please make sure to cite the PEMS04 dataset and STGODE model in your work.

## Credits

We would like to express our gratitude to the creators of the PEMS dataset and the developers of the STGODE model. This work is built upon their research and contributions:

- [PEMS Traffic Monitoring Dataset](http://pems.dot.ca.gov/)
- [STGODE](https://github.com/square-coder/STGODE)
