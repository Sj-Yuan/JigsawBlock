# Anomaly_detection_for_Texture_based_product

![UMSFAM_1](Anomaly_detection_for_Texture_based_product\src\img\UMSFAM_1.jpg)
![UMSFAM_1](Anomaly_detection_for_Texture_based_product\src\img\MFCSAM_1.jpg)

仓库为文章的模块代码，包括UMSFAM以及其他的src文件夹中的组成模块。
使用时，需要将代码加入使用的backbone网络中，并将网络的输出特征加入代码。

除了代码以外，我们提供了MTL AD Dataset的地址：
xxxxxxxxxxx

代码效果分别在MVTec AD dataset以及MTL AD Dataset中的Texture-based类别上进行测试，
# 测试结果
检测结果——CFA

| class   | CFA   | CFA_MFCSAM_layer2 | CFA_UMSFAM |
| ------- | ----- | ----------------- | ---------- |
| carpet  | 99.5  | 99.4              | 99.4       |
| grid    | 99.2  | 98.7              | **99.9**   |
| leather | 100   | 100               | 100        |
| tile    | 99.4  | 100               | **100**    |
| wood    | 99.2  | 99.4              | **99.4**   |
| avg.    | 99.46 | 99.5              | **99.74**  |



检测结果——Patchcore

| class   | Patchcore | Patchcore_MFCSAM_layer2 | Patchcore_UMSFAM |
| ------- | --------- | ----------------------- | ---------------- |
| carpet  | 97.8      | 98.5                    | **99.2**         |
| grid    | 96.4      | 98.4                    | 98               |
| leather | 100       | 100                     | 100              |
| tile    | 100       | 100                     | 100              |
| wood    | 99.2      | 99.1                    | 99.2             |
| avg.    | 98.68     | 99.2                    | **99.28**        |


