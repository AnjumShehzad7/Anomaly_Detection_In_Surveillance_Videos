# Anomaly Detection in Surveillance Videos

"Real-world Anomaly Detection in Surveillance Videos" using PyTorch. The model leverages a feature extractor and a classifier to detect anomalies in surveillance videos.

## Datasets

Download the dataset from the following [link](https://drive.google.com/file/d/18nlV4YjPM93o-SdnPQrvauMN_v-oizmZ/view?usp=sharing) and unzip it under your `$DATA_ROOT_DIR`.

```shell
/workspace/DATA/UCF-Crime/all_rgbs
```

Defines the location where datasets should be stored.

### Directory Structure
```shell
DATA/
    UCF-Crime/
        ../all_rgbs
            ../~.npy
        ../all_flows
            ../~.npy
    train_anomaly.txt
    train_normal.txt
    test_anomaly.txt
    test_normal.txt
```
This provides a structured directory tree for dataset organization.

---

### 4️⃣ **Training and Testing Instructions**
## Training and Testing

Run the following script to train and test the model:

```bash
python main.py
```
This section tells users how to run the training/testing script.

---

### 5️⃣ **Model Components Explanation**
### Model Components
The implementation includes the following key components:
- **Feature Extractor**: Extracts deep features from video frames.
- **Classifier**: Learns to distinguish normal and anomalous events.
- **Loss Function (MIL Loss)**: Optimized for anomaly detection using Multiple Instance Learning.
- **Dataset Loader**: Handles loading and preprocessing of normal and anomalous video frames.

## Results

| METHOD | DATASET | AUC |
|:--------:|:--------:|:--------:|
| Original paper (C3D two stream) | UCF-Crimes | 75.41 |
| [RTFM](https://arxiv.org/pdf/2101.10030.pdf) (I3D RGB) | UCF-Crimes | 84.03 |
| Ours Re-implementation (I3D two stream) | UCF-Crimes | 84.45 |

## Visualization

Below is a sample visualization of the anomaly detection predictions:

<table>
  <tr>
    <td><img alt="Visualization" src="./result.png" height="280" width="400" /></td>
  </tr>
</table>

Displays an image from the results.

---

### 8️⃣ **Dependencies Installation**

## Dependencies

Make sure you have the following dependencies installed:

```bash
pip install torch torchvision numpy opencv-python matplotlib tqdm
```
Lists the necessary Python libraries for running the project.

---

### 9️⃣ **Citation and Contribution**

## Citation
If you find this work useful, please cite the original paper and the RTFM reference.

For any issues, feel free to open an issue or contribute to this repository.
