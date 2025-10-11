# side-channel-attack-dataset
This repository contains a cache side-channel attack dataset based on stress-ng workloads.

Due to certain components being designed for specific operating systems, not all source files are included.
We have shared a portion of the dataset under different workload intensities:
three compressed files — low, medium, and high — correspond to data collected under varying load levels.

A data collection script for the Ubuntu platform is provided.
For details on attack-related file acquisition, please refer to the following dataset:
https://figshare.com/articles/dataset/DataSet/20528937/4.

We additionally apply a data augmentation pipeline to enrich the time-series dataset. The augmentation follows a simple image-based workflow inspired by CycleGAN (see https://github.com/junyanz/CycleGAN). Concretely, we convert time-series signals into images (Gramian angular difference field), perform image-to-image translation using CycleGAN to create augmented variants, and then transform the generated images back into time-series form. 

Note that our approach **only** uses CycleGAN-style translation on the intermediate image representations; we do not modify the dataset collection procedures. The image-based augmentation is intended to introduce plausible variations in temporal patterns while preserving overall signal characteristics after conversion back to time-series. For implementation details and training configurations, please refer to the official CycleGAN repository linked above.

If you have any questions, please contact me:cug-xinjun@qq.com
