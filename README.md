
**<h2 align="center">TransVOD: Underwater Cross-Domain Collaborative Spatial-Temporal Transformer Detector</h2>**

<details>   <summary>Fig</summary>   <div style="display: flex; justify-content: space-between;">     <img src="https://github.com/Anchor1566/TransUTD/blob/main/images/fig8a.png" alt="fig1" style="width: 48%;">     <img src="https://github.com/Anchor1566/TransUTD/blob/main/images/fig8b.png" alt="fig2" style="width: 48%;">   </div> </details>

## Main Results

### DUO

|  Model   | Backbone | Epoch | mAP  | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |                           Download                           |
| :------: | :------: | :---: | :--: | :-------------: | :-------------: | :------------: | :------------: | :------------: | :----------------------------------------------------------: |
| TransUTD | ResNet50 |  12   | 66.0 |      86.0       |      73.9       |      55.9      |      66.7      |      64.0      | [log](https://github.com/user-attachments/files/19787301/transutd.txt) / [config]() / [checkpoint]() |
| TransUTD | ResNet50 |  36   | 70.3 |      88.8       |      78.3       |      56.9      |      71.6      |      69.6      | [log](https://github.com/user-attachments/files/19787307/transutd-duo3x.txt) / [config]() / [checkpoint]() |

### UVID

|     Model      | Backbone | Epoch | mAP  | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> | Download                                                     |
| :------------: | :------: | :---: | :--: | :-------------: | :-------------: | :------------: | :------------: | :------------: | ------------------------------------------------------------ |
|    YOLO11*     |    -     |  72   | 56.6 |      78.5       |        -        |       -        |       -        |       -        | [log](https://github.com/user-attachments/files/19787602/yolo11-1.csv) / [checkpoint]() |
|      DINO      | ResNet50 |  12   | 52.3 |      76.4       |      58.4       |      29.1      |      35.2      |      55.3      | [log](https://github.com/user-attachments/files/19787326/DINO-UVID.log) / [checkpoint]() |
| Relation-DETR  | ResNet50 |  12   | 55.3 |      78.2       |      61.7       |      31.0      |      38.6      |      58.4      | [log](https://github.com/user-attachments/files/19787327/relation_training_UVID.log) / [checkpoint]() |
|    RT-DETR     | ResNet50 |  12   | 57.0 |      81.1       |      64.0       |      33.6      |      35.7      |      61.2      | [log](https://github.com/user-attachments/files/19787326/DINO-UVID.log) / [checkpoint](https://github.com/user-attachments/files/19787370/RT-DETR.txt) |
|    Co-DTER     | ResNet50 |  12   | 51.7 |      75.9       |      57.3       |      35.8      |      35.7      |      53.1      | [log](https://github.com/user-attachments/files/19787323/Co-DETR1.log) / [checkpoint]() |
|    GCC-Net     |  Swin-T  |  12   | 48.9 |      73.8       |      53.8       |      32.0      |      31.3      |      49.3      | [log](https://github.com/user-attachments/files/19787336/GCC-Net.log) / [checkpoint]() |
|  Dynamic YOLO  |    -     |  12   | 49.0 |      73.4       |      53.9       |      17.0      |      32.2      |      52.6      | [log](https://github.com/user-attachments/files/19787335/dynamic-yolo.log) / [checkpoint]() |
|     UDMDET     | ResNet50 |  12   | 46.1 |      73.2       |      49.7       |      21.3      |      30.4      |      48.5      | [log](https://github.com/user-attachments/files/19787333/UMDET.log) / [checkpoint]() |
| Boosting R-CNN | ResNet50 |  12   | 51.6 |      75.2       |      67.4       |      29.7      |      35.0      |      54.1      | [log](https://github.com/user-attachments/files/19787321/Boosting-R-CNN.log) / [checkpoint]() |
|    TransUTD    | ResNet50 |  12   | 58.9 |      82.8       |      66.2       |      35.3      |      37.2      |      62.7      | [log](https://github.com/user-attachments/files/19787371/TranUTD.txt) / [config]() / [checkpoint]() |
|   TransUTD*    | ResNet50 |  12   | 60.6 |      83.8       |      67.8       |      37.9      |      39.0      |      64.3      | [log](https://github.com/user-attachments/files/19787372/TransUTD-pre.txt) / [config]() / [checkpoint]() |

**Notes**:

- `*` means COCO-pretrained

## Peformance

**Difficult Conditions**

![difficult](https://github.com/Anchor1566/TransUTD/blob/main/images/fig11.jpg "difficult")

## UVID

**UVID** is a specialized underwater video object detection dataset created exclusively for research purposes. As the first dataset dedicated to this domain, UVID comprises 46,962 annotated frames and 191,699 object instances, representing five prominent underwater species: holothurians, urchins, scallops, starfish, and fish. The dataset is built upon the [KIS_MVK](https://github.com/quangtrungtruong/KIS_MVK) framework and includes underwater videos sourced from the internet, with special thanks to the contributors, as well as real-world underwater footage captured by our divers.

In the UVID dataset, each frame is annotated with a `transform` field, indicating whether the frame represents a motion transform compared to the previous one. This annotation uses the values `1` for motioned frames and `0` for unchanged frames. This annotation strategy leverages local temporal consistency.

![UVID](https://github.com/Anchor1566/TransUTD/blob/main/images/fig3.png "UVID")

## Installtion

The codebase is built on top of [RT-DETR](https://github.com/lyuwenyu/RT-DETR).

## Get started

## TODO

code will be available soon.😊
