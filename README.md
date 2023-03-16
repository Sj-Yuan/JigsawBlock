# Anomaly Detection of Industrial Products Considering both Texture and Shape Information

![jigano](https://github.com/DiagoAlaraviz/JigsawBlock/blob/main/src/img/model_3.png)

The repository is the code of article, including PFAM and other components in the src folder. When using it, you need to add the code to the backbone network.

In addition to the code, we provide the address of the [MTL AD Dataset](https://drive.google.com/drive/folders/1PdEcDyFPb3d7yb5uQCOrbi3fs3PyoilG).
The code effect is tested on MVTec AD dataset and MTL AD Dataset respectively.
# Result

## MVTec AD

CFA

| class      | CFA           | Ours         |
| ---------- | ------------- | ---------------- |
| carpet     | **99.5**/98.7 | 99.4/**99.0**    |
| grid       | 99.2/97.8     | **99.9/97.8**    |
| leather    | 100/99.1      | 100/**99.5**     |
| tile       | 99.4/95.8     | **100**/**96.1** |
| wood       | 99.7/94.8     | **100**/94.8     |
| bottle     | 100/98.6      | 100/98.6         |
| cable      | 99.8/**98.7** | **99.9**/98.6    |
| capsule    | 97.3/98.9     | **98.2**/98.9    |
| hazelnut   | 100/98.6      | 100/98.6         |
| metal_nut  | **100/98.8**  | 99.6/98.7        |
| pill       | 97.9/**98.6** | **98.7**/98.2    |
| screw      | 97.3/**99.0** | 97.3/98.9        |
| toothbrush | **100**/98.8  | 99.7/**98.9**    |
| transistor | 100/98.3      | 100/**98.4**     |
| zipper     | 99.6/98.6     | **99.7/98.8**    |
| avg.       | 99.3/98.2     | **99.5/98.25**   |

PatchCore

| class      | PatchCore           | Ours         |
| ---------- | ------------- | ---------------- |
| carpet     | 98.4/**98.8** | **99.2**/98.7 |
| grid       | 95.9/96.8     | **98/97.6** |
| leather    | 100/**99.1**  | 100/99 |
| tile       | 100/**96.1** | 100/95.4 |
| wood       | 98.9/93.4     | **99.2/95.1** |
| bottle     | 100/98.4      | 100/**98.8** |
| cable      | 99.0/**98.8** | **99.2**/98.7 |
| capsule    | **98.2**/98.8 | 97.4/**99.2** |
| hazelnut   | 100/98.7      | 100/**98.9** |
| metal_nut  | **99.4**/98.9 | 98.6/**99.3** |
| pill       | 92.4/**98** | **92.5**/97 |
| screw      | 96/98.9 | **96.2/99.5** |
| toothbrush | 93.3/98.8  | **100/99** |
| transistor | 100/98.1      | 97.3/97.8 |
| zipper     | **98.2**/98.3 | **96.4/99** |
| avg.       | 98/98 | **98.27/98.2** |

## MTL AD

CFA

| class     | CFA           | CFA_MFCSAM_layer4 | CFA_PFAM        |
| --------- | ------------- | ----------------- | ----------------- |
| ostrich   | 91.3/**86.7** | 91.8/85.5         | **93.2**/86.4     |
| lychee    | 87.7/93.9     | 89.1/93.2         | **93.6**/**94.8** |
| pearlfish | 79.5/91.3     | 79.2/90.7         | **82.8**/**91.8** |
| avg.      | 86.16/90.63   | 86.7/89.8         | **89.86**/**91**  |

PatchCore

| class     | Patchcore     | Patchcore_MFCSAM_layer4 | Patchcore_PFAM |
| --------- | ------------- | ----------------------- | ---------------- |
| ostrich   | 84.8/82.3     | 86.3/80.8               | **86.9**/80.8    |
| lychee    | **75.2**/89.4 | 73.5/89.8               | 73.9/**89.9**    |
| pearlfish | 75.3/87.5     | 80.6/**91**             | **81.2**/90.9    |
| avg.      | 78.4/86.4     | 79.7/87.2               | **80.6**/87.2    |

# Testing——CFA
## MVTec AD
![mvtec_result_shape](https://github.com/YShaoJiang/Anomaly_detection_for_Texture_based_product/blob/main/src/img/mvtec_shape_result.jpg)
![mvtec_result_shape_2](https://github.com/YShaoJiang/Anomaly_detection_for_Texture_based_product/blob/main/src/img/mvtec_shape_result_2.jpg)
![mvtec_result](https://github.com/YShaoJiang/Anomaly_detection_for_Texture_based_product/blob/main/src/img/mvtec_result.jpg)

## MTL AD
![mtl_result](https://github.com/YShaoJiang/Anomaly_detection_for_Texture_based_product/blob/main/src/img/mtl_result.jpg)
