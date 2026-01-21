# Modified Codes from nn-UNetv2


Reference: [nn-UNetv2](https://github.com/MIC-DKFZ/nnUNet).

I finally decided to directly borrow codes from nn-UNetv2 and make some adaptations. I have struggled for months on implementing nn-UNetv2's pipeline with [MONAI](https://github.com/Project-MONAI/MONAI), but still failed in achieving the same level of performance on a few anisotropic datasets like MSD Prostate. Some functionalities are pretty different in their implementation and hard to re-implement. Really really anxious about potentially critical demands on high DICE score from reviewers. I wished to figure out what's still wrong with my implementation (there should be not much left as it works well on the majority of datasets), but I am a bit nervous as my project has taken much longer than peers now (sigh...).

Quite appreciate the efforts done by MIC-DKFZ contributing to the community such a robust system. Please let me know if I should not have used the public code in this way.