# Anomaly_detection_for_Texture_based_product

![UMSFAM_2](https://github.com/YShaoJiang/Anomaly_detection_for_Texture_based_product/blob/main/src/img/UMSFAM_2.jpg)

仓库为文章的模块代码，包括UMSFAM以及其他的src文件夹中的组成模块。
使用时，需要将代码加入使用的backbone网络中，并将网络的输出特征加入代码。

除了代码以外，我们提供了MTL AD Dataset的地址：
xxxxxxxxxxx

代码效果分别在MVTec AD dataset以及MTL AD Dataset中的Texture-based类别上进行测试，
# 测试结果

## MVTec AD

检测结果——CFA

| class   | CFA         | CFA_MFCSAM_layer4 | CFA_UMSFAM          |
| ------- | ----------- | ----------------- | ------------------- |
| carpet  | 99.5/98.7   | 99/99.1           | **99.5**/99.1       |
| grid    | 99.2/97.8   | 99.6/98.5         | **99.6**/**98.6**   |
| leather | 100/99.1    | 100/99.4          | 100/**99.5**        |
| tile    | 99.4/95.8   | 100/96.4          | **100**/**97.1**    |
| wood    | 99.7/94.8   | 99.4/95.8         | **100**/**96.4**    |
| avg.    | 99.56/97.24 | 99.6/97.84        | **99.82**/**98.14** |

检测结果——Patchcore

| class   | Patchcore    | Patchcore_MFCSAM_layer4 | Patchcore_UMSFAM |
| ------- | ------------ | ----------------------- | ---------------- |
| carpet  | 97.8/98.8    | 99.1/98.8               | **99.2**/98.7    |
| grid    | 96.4/96.8    | 97.7/**98.3**           | 98/97.6          |
| leather | 100/**99.5** | 100/99.1                | 100/99           |
| tile    | 100/**96**   | 100/95.7                | 100/95.4         |
| wood    | 99.2/93.3    | **99.5**/93.8           | 99.2/**95.1**    |
| avg.    | 98.68/96.88  | 99.26/**97.16**         | **99.28**/97.14  |

## MTL AD

检测结果——CFA

| class     | CFA           | CFA_MFCSAM_layer4 | CFA_UMSFAM        |
| --------- | ------------- | ----------------- | ----------------- |
| ostrich   | 91.3/**86.7** | 91.8/85.5         | **93.2**/86.4     |
| lychee    | 87.7/93.9     | 89.1/93.2         | **93.6**/**94.8** |
| pearlfish | 79.5/91.3     | 79.2/90.7         | **82.8**/**91.8** |
| avg.      | 86.16/90.63   | 86.7/89.8         | **89.86**/**91**  |

检测结果——Patchcore

| class     | Patchcore     | Patchcore_MFCSAM_layer4 | Patchcore_UMSFAM |
| --------- | ------------- | ----------------------- | ---------------- |
| ostrich   | 84.8/82.3     | 86.3/80.8               | **86.9**/80.8    |
| lychee    | **75.2**/89.4 | 73.5/89.8               | 73.9/**89.9**    |
| pearlfish | 75.3/87.5     | 80.6/**91**             | **81.2**/90.9    |
| avg.      | 78.4/86.4     | 79.7/87.2               | **80.6**/87.2    |

# 测试过程——CFA
## MVTec AD
![mvtec_result](https://github.com/YShaoJiang/Anomaly_detection_for_Texture_based_product/blob/main/src/img/mvtec_result.jpg)

## MTL AD
![mtl_result](https://github.com/YShaoJiang/Anomaly_detection_for_Texture_based_product/blob/main/src/img/mtl_result.jpg)
