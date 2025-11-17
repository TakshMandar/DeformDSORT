DDSORT: Deformation-Aware DeepSORT for Multi-Object Tracking

This repository contains the implementation of DDSORT, an extension of the DeepSORT multi-object tracking framework. DDSORT incorporates a Gated Deformable Tracking (GDT) module that enhances appearance feature robustness under non-rigid motion, deformation, occlusion, and pose variation. The tracker follows the classical SORT/DeepSORT paradigm while improving the appearance embedding through deformable convolution and gated feature fusion.

Repository Structure
MAIN/GDT/

This directory contains all components of the GDT feature extractor, which replaces DeepSORT's standard CNN embedding.

deformable_conv_module.py
Implements the deformable convolutional layer (DCNv1). Generates learned spatial offsets and performs non-rigid sampling.

feature_extractor.py
Standard convolutional feature extractor based on the first 16 layers of the VGG-16 architecture.

gating_module.py
Learnable gating mechanism used to combine standard and deformable features via a sigmoid-based weighting module.

gdt_feature_extractor.py
Integrates deformable convolution and gating to produce the final deformation-aware descriptor.

gdt_wrapper.py
Interface wrapper for connecting the GDT module with the tracking pipeline

MAIN/sort/

Contains the full SORT/DeepSORT tracking pipeline, extended for DDSORT.

detection.py
Data structure and utilities for detection inputs.

iou_matching.py
Intersection-over-Union based association for fallback matching.

kalman_filter.py
Linear Kalman filter for motion prediction and uncertainty propagation.

linear_assignment.py
Implements the Hungarian algorithm for optimal association.

nn_matching.py
Computes cosine distance between GDT features and track appearance galleries.

preprocessing.py
Auxiliary utilities for preprocessing detections and embeddings.

track.py
Track state definition, feature gallery management, and lifecycle handling.

tracker.py
Core tracking logic: prediction, feature extraction, matching cascade, and track updates.

DDSORT.py

High-level module that integrates GDT with DeepSORT.
This file defines the final DDSORT tracker used for inference and evaluation.

configs/DDSORT.yaml

Configuration file containing all tracking parameters:

Feature fusion weight (λ)

Gating parameters

Descriptor dimensionality

Kalman filter settings

Matching thresholds

Maximum track age and initialization parameters

This YAML file governs the behavior of the entire DDSORT system
