# SLoRD

This is the official pytorch implementation of SLoRD.

# Pipeline
![image](https://github.com/AlexYouXin/SLoRD-VerSe/blob/main/pipeline.png)

1. Extracted basic contour descriptors via SVD using the 'contour_descriptors' folder.
2. Acquiring coarse predictions with files in the folder of 'the first stage'.
3. Refining coarse predictions with SLoRD in the folder of 'the second stage'.


# Detailed structures
![image](https://github.com/AlexYouXin/SLoRD-VerSe/blob/main/decoded_structure.png)


# Ackonwledgement
Part of codes are borrowed from other open-source github projects.

* https://github.com/MIC-DKFZ/MedNeXt
* https://github.com/MIC-DKFZ/nnUNet
* https://github.com/xieenze/PolarMask

# Citation
If you use our code or models in your work or find it is helpful, please cite the corresponding paper:  
```
@article{you2024slord,
  title={SLoRD: Structural Low-Rank Descriptors for Shape Consistency in Vertebrae Segmentation},
  author={You, Xin and Lou, Yixin and Zhang, Minghui and Zhang, Chuyan and Yang, Jie and Gu, Yun},
  journal={arXiv preprint arXiv:2407.08555},
  year={2024}
}
```
