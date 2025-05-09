## Quick Start
1. Download coco128 dataset
2. Prepare your own dataset
3. Augment the coco128 dataset with your own dataset
    * Randomly copy and paste your own dataset into the coco128 dataset
    ```bash
    python coco128/augment_new_classes.py
    ```
4. Fine-tune the model with augmented coco128 dataset
    * Update the data.yaml file with new classes
    * Run the training script
    ```bash
    sh train.sh
    ```
5. Evaluate the model
    * Prepare the validation set
    * Run the validation script
```bash
sh val.sh
```
6. Demo
    * Image
    ```bash
    python demo.py
    ```
    * Video
    ```bash
    python demo_video.py
```
