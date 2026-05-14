# ProstateLandmarkIdentification
Various models used for identifying prostate landmarks on surface-based ultrasound images of the prostate.

Training, testing, and validation packages can be found here for the following model types:
1. Image Patch Voting Variants 
2. Heatmap Variants
3. Detection Variants

## Image Patch Voting (IPV)
The original study for prostate volume estimation in surface-based ultrasound (SUS) images can be found:
- [Paper](https://doi.org/10.3390/app12031390)
- [Github](https://github.com/nurbalbayrak/prostate_volume_estimation)
- [Original code and data](https://drive.google.com/drive/folders/1uW2X7bTVSdHtlYxmkCbNwP-ra4X7QjC3)

While the original model setup worked incredibly well, there were improvements to be made. There was also an inconsistency between the paper and the code, namely the paper stated that 16 of the 18 ResNet18 layers were frozen during training, and only the final 2 layers were fine-tuned. It may be that I just misunderstood the original code, but it seemed like no layers were frozen.

Beside that, the code worked as expected but needed updating.

One of the primary drawbacks of the original IPV model was the amount of time it took to generate results from a single image. In an attempt to address this, alternative model backbones were used:
- Custom ResNet10 (more lightweight than the original).
- Custom ResNet14 (more lightweight than the original).
- Small CNN (more lightweight than the original).
- ResNet18 (original).
- ResNet34 (larger than the original, included out of interest).

However, during testing it was found that most of the time was spent generating the dataset and accummulating patch votes, and not on inference. Substantial speed gains were made using multiprocessing, but the grid-like nature of patch centre selection limited how much time-savings could be made; a grid step size of 10 just results in too many patches. It is for this reason that the other model types were investigated.  
