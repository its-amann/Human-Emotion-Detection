Emotion Detection Project
## Disclaimer

**Note:** The `main.ipynb` file preview on GitHub may not fully load or display all the content due to the size or complexity of the file. To view the complete code, please download the file and open it locally in a Jupyter Notebook environment. This will ensure you can access all the cells and outputs seamlessly.
---

# Table of Sections

| **Section** | **Description**                             |
|-------------|---------------------------------------------|
|  1           | Project Configuration                      |
|  2           | Dataset Loading and Preprocessing          |
|  3           | TF=Records Implemetation                   |
|  4           | Dataset Visualization                      |
|  5           | Model Architectures Overview               |
|  6           | LeNet Model                                |
|  7           | ResNet * *                                 |
|  8           | EfficientNetB *                            |
|  9           | Fine-Tuned Pretrained Model EfficientNetB  |
|  10          | VGG* *                                     |
|  11         | Grad-CAM Model for Explainability           |
|  12          | Vision Transformer (ViT) Model From Scratch|
|  13          | Vit Fine-tune                              |
|  14          | ONNX Model Export                          |
|  15          | Steps How to Run Application               |
|  16          | Conclusion and Future Work                 |

---
# Section *: Introduction

##  Project Overview
The primary objective of this project is to detect human emotions from facial images using deep learning models  We implement a diverse set of models, including CNN-based architectures, Vision Transformers (ViT), and explainability models such as Grad-CAM  The project focuses on achieving high accuracy while maintaining model interpretability 

##  Expected Outcomes
- **High Accuracy:** Achieving robust emotion classification 
- **Model Interpretability:** Visualizing model decisions using explainability techniques 
- **Scalability:** Deploying a model-ready system suitable for real-world environments 

##  Technology Stack
- **Frameworks:** TensorFlow, Keras
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Pretrained Models:** MobileNetV *, ResNet * *, Vision Transformer (ViT)
- **Development Environment:** Jupyter Notebook
- **Deployment** - FastAPi , Onnix

---

# Section  : Project Configuration

##   Library Imports
We import essential libraries for model development, evaluation, and visualization 

##   Hyperparameter Setup
- **Batch Size:** * *
- **Image Size:**  * * *x * * * pixels
- **Learning Rate:**  *  * **
- **Epochs:**  * *

##   Directory Structure
- **Train Directory:** Contains labeled training images 
- **Validation Directory:** Used for validation during training 
- **Test Directory:** Reserved for evaluation 

---

# Section   : Dataset Loading and Preprocessing

##   Dataset Source
The dataset comprises labeled facial emotion images, covering emotions such as happiness, sadness, anger, and surprise 

##   Loading Process
- Use TensorFlow's `image_dataset_from_directory()` to load and preprocess the dataset 

##   Preprocessing Steps
![alt text](image-* png)
- **Resizing and Rescaling:** Images resized to  * * *x * * * pixels and normalized 
- **Data Augmentation:** Apply random flips, rotations, and zooms 
- **Batching & Caching:** Ensure efficient data loading 

---
Here’s a detailed explanation for the **TFRecords implementation** section:

---

## TFRecords Implementation

TensorFlow TFRecords is a binary data format that provides an efficient way to store large datasets  This section details how TFRecords are created from the training and validation datasets 

###   Preparing Datasets
The code begins by unbatching the training and validation datasets  This step is necessary to access individual samples (images and labels) from batched datasets 

```python
training_dataset = (
    training_dataset
     unbatch()
)

validation_dataset = (
    validation_dataset
     unbatch()
)
```
- **` unbatch()`**: Breaks down a batched dataset into individual elements, enabling direct processing of each image-label pair 

---

###    Creating Serialized TFRecord Examples
A custom function, `create_example`, is implemented to serialize image-label pairs into a format compatible with TFRecords 
![image](https://github.com/user-attachments/assets/298ce961-99ce-4965-82e2-fbeaf3079de3)



```python
def create_example(image, label):
    bytes_feature = Feature(
        bytes_list=BytesList(value=[image])
    )

    int_feature = Feature(
        int * *_list=Int * *List(value=[label])
    )

    example = Example(
        features=Features(
            feature={
                'images': bytes_feature,
                'labels': int_feature,
            }
        )
    )

    return example SerializeToString()
```

#### Key Components:
*  **`Feature`**: A TensorFlow `Feature` object that defines how data is stored in the TFRecord 
   - **BytesList**: Encodes the raw image data (in bytes format) 
   - **Int * *List**: Encodes the label as an integer 
   
 *  **`Example`**: A TensorFlow object that encapsulates a single instance of data (e g , one image and its corresponding label) 
   - The `Example` is defined using a dictionary (`feature`) that maps feature names (like `images` and `labels`) to their respective serialized `Feature` objects 

 *  **Serialization**: 
   - **`SerializeToString()`**: Converts the `Example` into a serialized string format suitable for writing to a TFRecord file 

---

###    Writing TFRecords
The serialized examples can then be written to TFRecord files using a writer, typically as follows:

```python
with tf io TFRecordWriter('training tfrecord') as writer:
    for image, label in training_dataset:
        example = create_example(image, label)
        writer write(example)
```

#### Benefits of TFRecords:
- Efficient storage for large datasets 
- Optimized for TensorFlow pipelines, especially when combined with `tf data` 
- Enables faster training by reducing I/O overhead during data loading 

---
---


####    Directory Setup and Sharding
```python
!mkdir tfrecords

NUM_SHARDS = * *
PATH = 'tfrecords/shard_{: * *d} tfrecord'
```
- **Directory Creation**: Creates a directory named `tfrecords` to store the serialized TFRecord files  The script ignores errors if the directory already exists 
- **Sharding**: TFRecord files are divided into multiple shards (`NUM_SHARDS=* *`) to allow parallel access and improve I/O performance 

---

####    Encoding Images for TFRecords
```python
def encode_image(image, label):
    image = tf image convert_image_dtype(image, dtype=tf uint *)
    image = tf io encode_jpeg(image)
    return image, tf argmax(label)
```
- **Conversion to uint ***: Converts image pixel values to  *-bit integers for storage efficiency 
- **JPEG Encoding**: Encodes images as JPEG format to further compress the data 
- **Label Encoding**: Converts one-hot encoded labels to integer class indices using `tf argmax` 

```python
encoded_dataset = (
    training_dataset
     map(encode_image)
)
```
- **Dataset Mapping**: Applies the `encode_image` function to the dataset, converting each image and label into a format ready for TFRecords 

---

####    Writing Sharded TFRecord Files
```python
for shard_number in range(NUM_SHARDS):
    sharded_dataset = (
        encoded_dataset
         shard(NUM_SHARDS, shard_number)
         as_numpy_iterator()
    )

    with tf io TFRecordWriter(PATH format(shard_number)) as file_writer:
        for encoded_image, encoded_label in sharded_dataset:
            example = create_example(encoded_image, encoded_label)
            file_writer write(example)
```
- **Sharding**: Divides the dataset into `NUM_SHARDS` equal parts using ` shard(NUM_SHARDS, shard_number)` 
- **TFRecordWriter**: Writes serialized `Example` objects (created using the `create_example` function) to sharded TFRecord files 

---

####    Reading TFRecords
```python
recons_dataset = tf data TFRecordDataset(
    filenames=[PATH format(p) for p in range(NUM_SHARDS- *)]
)
```
- **`TFRecordDataset`**: Reads and loads data from specified TFRecord files into a TensorFlow dataset for further processing 
- **Shards**: The example reads all shards except the last two (`NUM_SHARDS- *`) 

---

####    Parsing TFRecords
```python
def parse_tfrecords(example):
    feature_description = {
        "images": tf io FixedLenFeature([], tf string),
        "labels": tf io FixedLenFeature([], tf int * *),
    }

    example = tf io parse_single_example(example, feature_description)
    example["images"] = tf image convert_image_dtype(
        tf io decode_jpeg(example["images"], channels= *), dtype=tf float * *
    )
    return example["images"], example["labels"]
```
- **Feature Description**: Specifies the format of serialized data:
  - `images`: A string (encoded as JPEG) 
  - `labels`: An integer (class index) 
- **Image Decoding**: JPEG images are decoded back to pixel values and converted to `float * *` 

```python
parsed_dataset = (
    recons_dataset
     map(parse_tfrecords)
     batch(CONFIGURATION["BATCH_SIZE"])
     prefetch(tf data AUTOTUNE)
)
```
- **Mapping**: Applies the `parse_tfrecords` function to deserialize data 
- **Batching**: Groups data into batches of size specified by the configuration 
- **Prefetching**: Optimizes pipeline execution by prefetching data for faster access 

---

####    Verifying Parsed Data
```python
for i in parsed_dataset take(*):
    print(i)
```
- **Validation**: Prints one batch of parsed images and labels to verify the correctness of the TFRecord processing pipeline 

---


# Section  *: Dataset Visualization

##  * * Sample Visualization
- Visualize dataset samples using Matplotlib 

##  *  * Labels Inspection
- ![alt text](image png)

##  *  * Data Balance Check
- Display class distribution using Seaborn bar plots 

---

# Section  *: Model Architectures Overview


---

## Model Architectures Overview

### *  LeNet-inspired Convolutional Neural Network (CNN)

This model is a custom implementation inspired by the classic LeNet architecture for image classification  It includes modern improvements such as batch normalization, dropout, and L * regularization to improve training stability and reduce overfitting 

#### **Model Summary:**
- **Input Shape:** (None, None,  *) – RGB images of variable size 
- **Layers Overview:**
  - **Resizing and Rescaling:** Images are resized to  * * *x * * * and normalized to [ *, *] 
  - **Convolutional Layers:** 
    - First Conv *D layer with  * filters,  *x * kernel, ReLU activation, and L * regularization 
    - Batch Normalization for stable training 
    - MaxPooling *D layer for downsampling 
    - Dropout (rate  *  *) to reduce overfitting 
  - **Second Convolutional Block:** 
    - Conv *D layer with * * filters,  *x * kernel, ReLU activation, and L * regularization 
    - Batch Normalization and MaxPooling *D for downsampling 
  - **Fully Connected Layers:** 
    - Flatten layer to reshape the output from the convolutional layers 
    - Dense layer with * * * * units, ReLU activation, and L * regularization 
    - Batch Normalization and Dropout (rate  *  *) 
    - Another Dense layer with  ** * units and ReLU activation 
    - Final Dense layer with softmax activation for classification into  * classes: "angry," "happy," "sad" 

#### **Summary Table:**
| Layer Type              | Output Shape  | Parameters  |
|------------------------|----------------|--------------|
| Conv *D + BatchNorm      | ( * * *,  * * *,  *) | * * * +  * *    |
| MaxPooling *D + Dropout | (* * *, * * *,  *) |  *           |
| Conv *D + BatchNorm      | (* * *, * * *, * *)|  * * * +  * *    |
| MaxPooling *D           | ( * *,  * *, * *)  |  *           |
| Flatten                | ( ** * * *)       |  *           |
| Dense (* * * *)           | (* * * *)        |  * *, * **,* * *  |
| BatchNorm + Dropout    | (* * * *)        |  *, * * *       |
| Dense ( ** *)            | ( ** *)         |  * * *, * * *     |
| BatchNorm              | ( ** *)         |  *, * * *       |
| Dense ( * - softmax)    | ( *)           | *, * * *       |
| **Total Parameters:**  | ** * *, ** *, * * ***|              |

---
### Model Architectures Overview: ResNet * *

#### *  **Residual Block**
The building block of the ResNet * * architecture is the `ResidualBlock`, which facilitates efficient feature learning by introducing skip connections 

```python
class ResidualBlock(Layer):
    def __init__(self, n_channels, n_strides=*):
        super(ResidualBlock, self) __init__(name="res_block")
        self dotted = (n_strides != *)

        self custom_conv_* = CustomConv *D(n_channels,  *, n_strides, padding="same")
        self custom_conv_ * = CustomConv *D(n_channels,  *, *, padding="same")
        self activation = Activation("relu")

        if self dotted:
            self custom_conv_ * = CustomConv *D(n_channels, *, n_strides)

    def call(self, input, training):
        x = self custom_conv_*(input, training=training)
        x = self custom_conv_ *(x, training=training)

        if self dotted:
            x_add = self custom_conv_ *(input, training=training)
            x_add = Add()([x, x_add])
        else:
            x_add = Add()([x, input])

        return self activation(x_add)
```

##### **Explanation**:
- **Skip Connections**:
  - The input is added back to the output of the convolutional layers (shortcut path) 
  - If the input and output shapes differ (indicated by `self dotted`), a *x* convolution (`self custom_conv_ *`) is applied to match the dimensions 
- **Components**:
  - Two  *x * convolutional layers 
  - ReLU activation function 
  - Optional *x* convolution (for dimensionality matching) 

---

####  *  **Complete ResNet * * Architecture**
The ResNet * * model stacks multiple `ResidualBlock` layers organized in stages 

```python
class ResNet * *(Model):
    def __init__(self):
        super(ResNet * *, self) __init__(name="resnet_ * *")
        self conv_* = CustomConv *D( * *,  *,  *, padding="same")
        self max_pool = MaxPooling *D( *,  *)

        self conv_ *_* = ResidualBlock( * *)
        self conv_ *_ * = ResidualBlock( * *)
        self conv_ *_ * = ResidualBlock( * *)

        self conv_ *_* = ResidualBlock(* * *,  *)
        self conv_ *_ * = ResidualBlock(* * *)
        self conv_ *_ * = ResidualBlock(* * *)
        self conv_ *_ * = ResidualBlock(* * *)

        self conv_ *_* = ResidualBlock( * * *,  *)
        self conv_ *_ * = ResidualBlock( * * *)
        self conv_ *_ * = ResidualBlock( * * *)
        self conv_ *_ * = ResidualBlock( * * *)
        self conv_ *_ * = ResidualBlock( * * *)
        self conv_ *_ * = ResidualBlock( * * *)

        self conv_ *_* = ResidualBlock( ** *,  *)
        self conv_ *_ * = ResidualBlock( ** *)
        self conv_ *_ * = ResidualBlock( ** *)

        self global_pool = GlobalAveragePooling *D()
        self fc_ * = Dense(CONFIGURATION["NUM_CLASSES"], activation="softmax")

    def call(self, x, training=True):
        x = self conv_*(x)
        x = self max_pool(x)

        x = self conv_ *_*(x, training=training)
        x = self conv_ *_ *(x, training=training)
        x = self conv_ *_ *(x, training=training)

        x = self conv_ *_*(x, training=training)
        x = self conv_ *_ *(x, training=training)
        x = self conv_ *_ *(x, training=training)
        x = self conv_ *_ *(x, training=training)

        x = self conv_ *_*(x, training=training)
        x = self conv_ *_ *(x, training=training)
        x = self conv_ *_ *(x, training=training)
        x = self conv_ *_ *(x, training=training)
        x = self conv_ *_ *(x, training=training)
        x = self conv_ *_ *(x, training=training)

        x = self conv_ *_*(x, training=training)
        x = self conv_ *_ *(x, training=training)
        x = self conv_ *_ *(x, training=training)

        x = self global_pool(x)
        return self fc_ *(x)
```

##### **Explanation**:
- **Input Layer**:
  - Starts with a  *x * convolutional layer followed by max pooling for initial feature extraction 
- **Residual Blocks**:
  - Organized into  * stages:
    - Stage *:  * * filters,  * residual blocks 
    - Stage  *: * * * filters,  * residual blocks (stride of  * for downsampling in the first block) 
    - Stage  *:  * * * filters,  * residual blocks 
    - Stage  *:  ** * filters,  * residual blocks 
- **Global Pooling**:
  - Reduces the spatial dimensions to a single vector per filter for classification 
- **Fully Connected Layer**:
  - Final dense layer with `softmax` activation for multi-class classification 

---

####  *  **Model Summary**
```python
resnet_ * * = ResNet * *()
resnet_ * *(tf zeros([*,  * * *,  * * *,  *]))
resnet_ * * summary()
```

**Output**:
| Layer Type                 | Output Shape  | Parameters |
|----------------------------|---------------|------------|
| CustomConv *D               | ?             |  *, * * *      |
| MaxPooling *D               | ?             |  *          |
| ResidualBlock (x *)         | ?             |  * *, * * *     |
| ResidualBlock (x *)         | ?             |  * **, * * *    |
| ResidualBlock (x *)         | ?             |  *, * * *, ** *  |
| GlobalAveragePooling *D     | ?             |  *          |
| Dense                      | ?             | *, * * *      |
| **Total Parameters**       | ** **, ***, * * ***|            |

---
![image](https://github.com/user-attachments/assets/73542207-a28a-44ea-8e74-934e94f66da4)


---
### Model Architectures Overview: Pretrained Model Using EfficientNetB *

#### **Pretrained Backbone**
The model leverages the `EfficientNetB *` architecture as a feature extractor, using pretrained weights from ImageNet 

```python
backbone = tf keras applications efficientnet EfficientNetB *(
    include_top=False,
    weights="imagenet",
    input_shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"],  *)
)
backbone trainable = False
```

##### **Explanation**:
*  **EfficientNetB * Backbone**:
   - `include_top=False`: Removes the fully connected layers from the pretrained EfficientNetB * 
   - `weights="imagenet"`: Loads the pretrained weights from the ImageNet dataset 
   - `input_shape`: Defines the input dimensions to match the dataset 
 *  **Frozen Backbone**:
   - `backbone trainable = False`: Freezes the backbone to retain pretrained features and prevent further training of these layers 

---

#### **Custom Classification Head**
A classification head is added on top of the pretrained backbone to adapt it to the target task 

```python
pretrained_model = tf keras Sequential([
    Input(shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"],  *)),
    backbone,
    GlobalAveragePooling *D(),
    Dense(CONFIGURATION["N_DENSE_*"], activation="relu"),
    BatchNormalization(),
    Dense(CONFIGURATION["N_DENSE_ *"], activation="relu"),
    Dense(CONFIGURATION["NUM_CLASSES"], activation="softmax"),
])
```

##### **Explanation**:
*  **Global Average Pooling**:
   - Reduces the spatial dimensions of the feature map from the backbone to a single vector per channel 
 *  **Fully Connected Layers**:
   - **Dense Layers**:
     - First dense layer (`CONFIGURATION["N_DENSE_*"]` units) with ReLU activation 
     - Second dense layer (`CONFIGURATION["N_DENSE_ *"]` units) with ReLU activation 
   - **Batch Normalization**:
     - Stabilizes the learning process and improves convergence 
 *  **Output Layer**:
   - Final dense layer with `CONFIGURATION["NUM_CLASSES"]` units and softmax activation for multi-class classification 

---

#### **Model Summary**
The model's architecture can be summarized as:

| Layer (type)                  | Output Shape      | Parameters   |
|-------------------------------|-------------------|--------------|
| EfficientNetB * (Functional)   | (None,  *,  *, * * * *) | * *, * * *, * * *   |
| GlobalAveragePooling *D        | (None, * * * *)      |  *            |
| Dense (* * * * units)            | (None, * * * *)      | *, * * *, * * *    |
| BatchNormalization            | (None, * * * *)      |  *, * * *        |
| Dense ( ** * units)             | (None,  ** *)       |  * * *, * * *      |
| Dense (NUM_CLASSES)           | (None,  *)         | *, * * *        |

**Total Parameters**:  * *, * * *, * * *  
**Trainable Parameters**:  *, * * *, ** *  
**Non-trainable Parameters**: * *, * * *, * **  

---

#### **Loss Function and Metrics**
```python
loss_function = CategoricalCrossentropy()
metrics = [CategoricalAccuracy(name="accuracy")]
```
- **Loss Function**:
  - `CategoricalCrossentropy`: Computes the cross-entropy loss for multi-class classification tasks 
- **Metrics**:
  - `CategoricalAccuracy`: Measures the accuracy for multi-class classification 

---

![image](https://github.com/user-attachments/assets/47fc208e-89f4-4eb3-a76a-970f11da5369)


### Model Architectures Overview: Fine-Tuned Pretrained Model Using EfficientNetB *

#### ***  Fine-Tuning EfficientNetB ***
The previously frozen EfficientNetB * backbone is now unfrozen to allow end-to-end fine-tuning of the entire model 

```python
backbone trainable = True
```

- **Unfreezing the Backbone**:
  - By setting `backbone trainable = True`, the model allows gradients to propagate through all layers of EfficientNetB * during training 
  - This enables the backbone to adapt its features to the specific target dataset 

---

#### ** *  Custom Classification Head**
The classification head remains the same, built on top of the pretrained backbone 

```python
input = Input(shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"],  *))
x = backbone(input, training=False)
x = GlobalAveragePooling *D()(x)
x = Dense(CONFIGURATION["N_DENSE_*"], activation="relu")(x)
x = BatchNormalization()(x)
x = Dense(CONFIGURATION["N_DENSE_ *"], activation="relu")(x)
output = Dense(CONFIGURATION["NUM_CLASSES"], activation="softmax")(x)

finetuned_model = Model(input, output)
```

- **Components**:
  - Input Layer: Accepts images of shape `(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"],  *)` 
  - **Global Average Pooling**: Reduces spatial dimensions to a single vector per channel 
  - **Dense Layers**:
    - First dense layer: `CONFIGURATION["N_DENSE_*"]` units, ReLU activation 
    - Second dense layer: `CONFIGURATION["N_DENSE_ *"]` units, ReLU activation 
  - **Batch Normalization**: Normalizes intermediate activations for better training stability 
  - **Output Layer**:
    - Dense layer with `CONFIGURATION["NUM_CLASSES"]` units and softmax activation for multi-class classification 

---

#### ** *  Model Summary**
The fine-tuned model is summarized as follows:

| Layer (type)                  | Output Shape      | Parameters   |
|-------------------------------|-------------------|--------------|
| InputLayer                    | (None,  * * *,  * * *,  *) |  *            |
| EfficientNetB * (Functional)   | (None,  *,  *, * * * *)  | * *, * * *, * * *   |
| GlobalAveragePooling *D        | (None, * * * *)      |  *            |
| Dense (* * * * units)            | (None, * * * *)      | *, * * *, * * *    |
| BatchNormalization            | (None, * * * *)      |  *, * * *        |
| Dense ( ** * units)             | (None,  ** *)       |  * * *, * * *      |
| Dense (NUM_CLASSES)           | (None,  *)         | *, * * *        |

**Total Parameters**:  * *, * * *, * * *  
**Trainable Parameters**: * *, ** *, * * *  
**Non-trainable Parameters**: * * *, * * *  

---

#### ** *  Compilation and Training**
The model is compiled with a categorical cross-entropy loss function and accuracy metric  It is trained for  * epochs 

```python
finetuned_model compile(
    optimizer=Adam(learning_rate=CONFIGURATION["LEARNING_RATE"] / * * *),
    loss=CategoricalCrossentropy(),
    metrics=[CategoricalAccuracy(name="accuracy")],
)

history = finetuned_model fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs= *,
    verbose=*,
)
```

- **Loss Function**:
  - `CategoricalCrossentropy`: Measures the cross-entropy loss for multi-class classification 
- **Optimizer**:
  - `Adam`: Adaptive optimization algorithm with a learning rate of `CONFIGURATION["LEARNING_RATE"] / * * *` 
- **Metrics**:
  - `CategoricalAccuracy`: Evaluates classification accuracy 
- **Training**:
  - Trains for  * epochs on the training dataset and evaluates on the validation dataset 

---

#### ** *  Training Progress**
The training log shows the accuracy and loss over epochs:

| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------|-------------------|---------------------|---------------|-----------------|
| *     |  *  * * * *            |  *  * * * *              | *  * ** *        | *  ** * *          |
|  *     |  *  * * **            |  *  * * * *              |  *  * * * *        |  *  * * * *          |
|  *     |  *  * * * *            |  *  * * * *              |  *  * * * *        |  *  ** * *          |
|  *     |  *  * * * *            |  *  * * * *              |  *  * * * *        |  *  * * * *          |
|  *     |  *  * * * *            |  *  * * * *              |  *  * * * *        |  *  ** * *          |

---
## Results
![image](https://github.com/user-attachments/assets/8908c9d8-b08a-4f1d-b1b2-0d306a0ee84e)


### VGG* * Model and Feature Map Visualization

#### ***  VGG* * Backbone**
The VGG* * model is initialized as a feature extractor 

```python
vgg_backbone = tf keras applications vgg* * VGG* *(
    include_top=False,
    weights=None,
    input_shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"],  *)
)
vgg_backbone summary()
```

- **Explanation**:
  - **`include_top=False`**: Removes the fully connected layers to extract only the convolutional and pooling layers 
  - **Weights**: Set to `None`, indicating the model is not pretrained 
  - **Input Shape**: Configured to match the dataset's image dimensions 

##### **Model Summary**:
| Layer (type)                 | Output Shape         | Parameters   |
|------------------------------|----------------------|--------------|
| InputLayer                   | (None,  * * *,  * * *,  *) |  *            |
| block*_conv* (Conv *D)        | (None,  * * *,  * * *,  * *) | *, * * *        |
| block*_conv * (Conv *D)        | (None,  * * *,  * * *,  * *) |  * *, * * *       |
| block*_pool (MaxPooling *D)   | (None, * * *, * * *,  * *) |  *            |
| block *_conv* (Conv *D)        | (None, * * *, * * *, * * *)|  * *, * * *       |
|                              |                      |              |
| block *_pool (MaxPooling *D)   | (None,  *,  *,  ** *)    |  *            |

**Total Parameters**: * *, ** *, * * *  
**Trainable Parameters**: * *, ** *, * * *  

---

#### ** *  Feature Map Extraction**
A custom model is built to output feature maps from all convolutional layers in the VGG* * backbone 

```python
def is_conv(layer_name):
    if 'conv' in layer_name:
        return True
    else:
        return False

feature_maps = [
    layer output for layer in vgg_backbone layers[*:] if is_conv(layer name)
]
feature_map_model = Model(
    inputs=vgg_backbone input,
    outputs=feature_maps
)
feature_map_model summary()
```

- **Explanation**:
  - **`is_conv`**: Helper function to identify convolutional layers 
  - **Feature Map Model**:
    - Takes the same input as the VGG* * model 
    - Outputs feature maps from all convolutional layers 

---

#### ** *  Feature Map Visualization**
A single test image is processed to extract feature maps from the custom feature map model 

```python
test_image = cv * imread("path_to_image")
test_image = cv * resize(test_image, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))
im = tf constant(test_image, dtype=tf float * *)
im = tf expand_dims(im, axis= *)

f_maps = feature_map_model predict(im)
```

- **Preprocessing**:
  - Image is read and resized to match the model's input dimensions 
  - Converted to a tensor and batch dimension is added 
- **Feature Map Extraction**:
  - The `feature_map_model` is used to predict feature maps for the input image 

---

#### ** *  Visualizing Feature Maps**
Each feature map is visualized using `matplotlib` 
![image](https://github.com/user-attachments/assets/e8a7fdce-db74-4843-8c24-b1c18ac6b1e7)

```python
import matplotlib pyplot as plt
import numpy as np

max_channels =  * *  # Limit the number of channels displayed
for i in range(len(f_maps)):
    f_size = f_maps[i] shape[*]
    n_channels = min(f_maps[i] shape[ *], max_channels)
    joint_maps = np ones((f_size, f_size * n_channels))
    for j in range(n_channels):
        joint_maps[:, f_size * j:f_size * (j + *)] = f_maps[i][ *,    , j]

    plt figure(figsize=(* *, * *))
    plt imshow(joint_maps, cmap="viridis")
    plt title(f"Feature Map {i + *} | Channels: {n_channels}", fontsize=* *)
    plt axis("off")
    plt show()
```

- **Explanation**:
  - Limits the number of channels visualized to `max_channels` 
  - Combines feature maps from all channels into a single image for each convolutional layer 
  - Displays the combined feature map using a color map (`viridis`) 

---
### Grad-CAM Implementation Explanation

#### Step *: Initialize Pretrained Backbone
- The EfficientNetB * model is used as the backbone for feature extraction  
- `include_top=False` ensures that the fully connected layers at the top are excluded 
- The input shape of the model is specified using `CONFIGURATION["IM_SIZE"]` 

```python
backbone = tf keras applications efficientnet EfficientNetB *(
    include_top=False,
    weights='imagenet',
    input_shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"],  *),
)
backbone trainable = False
```

#### Step  *: Model Construction
- The pretrained backbone is augmented with global average pooling and several dense layers for classification 
- Final activation uses the softmax function for multi-class classification 

```python
x = backbone output
x = GlobalAveragePooling *D()(x)
x = Dense(CONFIGURATION["N_DENSE_*"], activation='relu')(x)
x = Dense(CONFIGURATION["N_DENSE_ *"], activation='relu')(x)
output = Dense(CONFIGURATION["NUM_CLASSES"], activation='softmax')(x)
pretrained_model = Model(backbone inputs, output)
```

#### Step  *: Preprocess Input Image
- The test image is resized to match the input dimensions of the model 
- The image tensor is expanded to add the batch dimension and cast to `tf float * *` 

```python
test_image = cv * imread(img_path)
test_image = cv * resize(test_image, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))
img_array = tf expand_dims(test_image, axis= *)
```

#### Step  *: Obtain Predictions
- Predictions are made using the constructed model, and the class with the highest probability is selected 

```python
preds = pretrained_model predict(img_array)
top_pred_index = np argmax(preds[ *])
```

#### Step  *: Grad-CAM Implementation
- **Last Convolutional Layer**:
  - Extract the last convolutional layer output 
  - Use the `tf GradientTape` to compute gradients of the top predicted class with respect to this layer's output 

```python
last_conv_layer_name = "top_activation"
last_conv_layer = pretrained_model get_layer(last_conv_layer_name)
last_conv_layer_model = Model(pretrained_model inputs, last_conv_layer output)

with tf GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(img_array)
    preds = pretrained_model(img_array)
    top_pred_index = tf argmax(preds[ *])
    top_class_channel = preds[:, top_pred_index]
    grads = tape gradient(top_class_channel, last_conv_layer_output)
```

#### Step  *: Create Heatmap
- Compute the Grad-CAM heatmap by averaging gradients across channels 
- Apply ReLU to remove negative values 
- Normalize the heatmap for visualization 

```python
pooled_grads = tf reduce_mean(grads, axis=( *, *,  *))
heatmap = tf reduce_sum(tf multiply(pooled_grads, last_conv_layer_output), axis=-*)
heatmap = tf nn relu(heatmap)
heatmap = tf image resize(heatmap, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))
```

#### Step  *: Overlay Heatmap on Image
- The heatmap is resized to match the image dimensions and overlaid on the input image for visualizing regions that contribute to the prediction 

```python
resized_heatmap = cv * resize(heatmap numpy(), ( * * *,  * * *))
plt matshow(resized_heatmap + test_image[ *, :, :,  *] /  * * *)
```

#### Step  *: Final Visualization
- The overlayed heatmap visually highlights the regions most influential to the model's prediction, allowing interpretability of results  
![image](https://github.com/user-attachments/assets/16e79d44-84cb-40cd-a98c-a56472caca20)


---

### Summary
The Grad-CAM implementation effectively leverages gradients and feature maps from the last convolutional layer to identify which parts of the image contributed to the model's decision  This is crucial for model explainability, especially in sensitive applications like emotion detection 


---

## Vision Transformer (ViT) - From Scratch

This section explains the implementation of a Vision Transformer (ViT) model from scratch using TensorFlow and Keras  Vision Transformers are powerful deep learning architectures based on the transformer mechanism, originally designed for NLP tasks but adapted for computer vision tasks 

### *  **Image Patch Extraction**

The first step involves dividing the input image into smaller patches, which act as input tokens for the transformer 

```python
patches = tf image extract_patches(
    images=tf expand_dims(test_image, axis= *),
    sizes=[*, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], *],
    strides=[*, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], *],
    rates=[*, *, *, *],
    padding='VALID'
)
patches = tf reshape(patches, (patches shape[ *], -*, patches shape[-*]))
```

- **Purpose:** Extracts non-overlapping patches from the input image 
- **Result:** Converts an input image into a sequence of patches, where each patch is treated as a token 

---

###  *  **Patch Encoder Layer**

This layer encodes the patches into embeddings, which are used as inputs to the transformer 

```python
class PatchEncoder(Layer):
    def __init__(self, N_PATCHES, HIDDEN_SIZE):
        super(PatchEncoder, self) __init__(name="patch_encoder")
        self linear_projection = Dense(HIDDEN_SIZE)
        self positional_embedding = Embedding(N_PATCHES, HIDDEN_SIZE)

    def call(self, x):
        patches = tf image extract_patches(
            images=x,
            sizes=[*, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], *],
            strides=[*, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], *],
            rates=[*, *, *, *],
            padding='VALID'
        )
        patches = tf reshape(patches, (tf shape(patches)[ *], -*, patches shape[-*]))
        positions = tf range(start= *, limit=self N_PATCHES, delta=*)
        return self linear_projection(patches) + self positional_embedding(positions)
```

- **Components:**
  - **Linear Projection:** Projects the patch into a fixed-dimensional embedding 
  - **Positional Embedding:** Adds positional information to each patch 
- **Purpose:** Converts image patches into feature embeddings 

---
![image](https://github.com/user-attachments/assets/207b7659-5d39-40e3-9a43-1028684fd04e)


###  *  **Transformer Encoder Layer**

This custom layer implements a single transformer encoder block 

```python
class TransformerEncoder(Layer):
    def __init__(self, N_HEADS, HIDDEN_SIZE):
        super(TransformerEncoder, self) __init__(name="transformer_encoder")
        self layer_norm_* = LayerNormalization()
        self layer_norm_ * = LayerNormalization()
        self multi_head_att = MultiHeadAttention(num_heads=N_HEADS, key_dim=HIDDEN_SIZE)
        self dense_* = Dense(HIDDEN_SIZE, activation=tf nn gelu)
        self dense_ * = Dense(HIDDEN_SIZE, activation=tf nn gelu)

    def call(self, inputs):
        x = self layer_norm_*(inputs)
        x = self multi_head_att(x, x)
        x = Add()([x, inputs])

        y = self layer_norm_ *(x)
        y = self dense_*(y)
        y = self dense_ *(y)
        return Add()([x, y])
```

- **Components:**
  - **Layer Normalization:** Normalizes inputs to stabilize training 
  - **Multi-Head Attention:** Allows the model to attend to different parts of the input sequence 
  - **Feedforward Network:** Applies two dense layers with a GELU activation 
- **Purpose:** Processes the sequence of patches and captures interdependencies 

---

###  *  **ViT Model Architecture**

This combines the patch encoder and transformer layers to form the full ViT architecture 

```python
class ViT(Model):
    def __init__(self, N_PATCHES, HIDDEN_SIZE, N_LAYERS, N_HEADS, N_CLASSES):
        super(ViT, self) __init__(name="vision_transformer")
        self patch_encoder = PatchEncoder(N_PATCHES, HIDDEN_SIZE)
        self transformer_encoders = [TransformerEncoder(N_HEADS, HIDDEN_SIZE) for _ in range(N_LAYERS)]
        self dense_* = Dense(* * *, activation=tf nn gelu)
        self dense_ * = Dense( * *, activation=tf nn gelu)
        self classifier = Dense(N_CLASSES, activation="softmax")

    def call(self, inputs):
        x = self patch_encoder(inputs)
        for encoder in self transformer_encoders:
            x = encoder(x)
        x = Flatten()(x)
        x = self dense_*(x)
        x = self dense_ *(x)
        return self classifier(x)
```

- **Patch Encoder:** Converts images into patch embeddings 
- **Transformer Encoders:** Stacks multiple transformer layers for hierarchical feature extraction 
- **Dense Layers:** Final layers for classification 
- **Classifier:** Outputs probabilities for each class 

---

###  *  **Model Compilation and Training**

The ViT model is compiled and trained on a dataset 

```python
vit = ViT(
    N_PATCHES=CONFIGURATION["N_PATCHES"],
    HIDDEN_SIZE=CONFIGURATION["HIDDEN_SIZE"],
    N_LAYERS=CONFIGURATION["N_LAYERS"],
    N_HEADS=CONFIGURATION["N_HEADS"],
    N_CLASSES=CONFIGURATION["NUM_CLASSES"]
)
vit compile(
    optimizer=Adam(learning_rate=CONFIGURATION["LEARNING_RATE"]),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
history = vit fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=CONFIGURATION["EPOCHS"]
)
```

- **Purpose:** Trains the ViT model using categorical cross-entropy and an Adam optimizer 

---

This implementation builds the Vision Transformer from scratch, extracting patches from images, embedding them, and processing them using transformer layers  The model is modular, allowing customization of patch size, embedding dimensions, number of layers, and attention heads 

--- 

Here's a detailed explanation of the fine-tuning process for pre-trained Vision Transformer (ViT) models that you can copy-paste into your GitHub README file:

---

## Fine-Tuning a Pre-Trained Vision Transformer (ViT)

Fine-tuning involves adapting a pre-trained model to a specific dataset or task  In this example, we fine-tune a pre-trained Vision Transformer (ViT) for image classification tasks 

---

### *  **Load Pre-Trained ViT Model**

We load a pre-trained Vision Transformer model using the `transformers` library 

```python
from transformers import TFViTModel

# Load the pre-trained ViT model
base_model = TFViTModel from_pretrained("google/vit-base-patch* *- * * *-in **k")

# Prepare a dummy input to check the model's output
dummy = tf random uniform(( *,  * * *,  * * *,  *), dtype=tf float * *)
transposed_dummy = tf transpose(dummy, perm=[ *,  *, *,  *])  # ViT expects channels-first format
test_out = base_model(pixel_values=transposed_dummy)

print(test_out last_hidden_state shape)
```

- **Model Source:** The `google/vit-base-patch* *- * * *-in **k` model is pre-trained on ImageNet- **k 
- **Output Shape:** The `last_hidden_state` represents the feature embeddings extracted by the model 

---

###  *  **Integrate Pre-Trained Model in Keras**

We wrap the pre-trained ViT model in a custom Keras layer for seamless integration with a fine-tuned architecture 

```python
class ViTModelLayer(tf keras layers Layer):
    def __init__(self, vit_model, **kwargs):
        super() __init__(**kwargs)
        self vit_model = vit_model

    def call(self, inputs):
        # Transpose inputs to channels-first format
        inputs = tf transpose(inputs, [ *,  *, *,  *])
        outputs = self vit_model(pixel_values=inputs)
        return outputs last_hidden_state
```

- **Purpose:** Allows the pre-trained model to process inputs in TensorFlow/Keras pipelines 
- **Input Transformation:** Adjusts input dimensions to match the format expected by the ViT model 

---

###  *  **Build the Fine-Tuning Model**

The fine-tuning model uses the ViT pre-trained backbone and adds custom layers for classification 

```python
from tensorflow keras layers import Dense, Input

input_layer = Input(shape=( * * *,  * * *,  *))
preprocessing_layer = Sequential([
    Resizing( * * *,  * * *),
    Rescaling(*  * /  * * *)
])(input_layer)

# Pass through the ViT model
vit_layer = ViTModelLayer(vit_model=base_model)(preprocessing_layer)

# Extract features and add a classification head
feature_layer = tf keras layers GlobalAveragePooling*D()(vit_layer)
output_layer = Dense(CONFIGURATION["NUM_CLASSES"], activation="softmax")(feature_layer)

fine_tuned_model = Model(inputs=input_layer, outputs=output_layer)

fine_tuned_model summary()
```

- **Preprocessing Layer:** Resizes and rescales input images for consistency with the pre-trained model 
- **Feature Layer:** Extracts the global features from the ViT model's output 
- **Output Layer:** A dense layer with a softmax activation predicts class probabilities 

---

###  *  **Compile the Model**

The fine-tuned model is compiled with an appropriate optimizer, loss function, and metrics 

```python
fine_tuned_model compile(
    optimizer=Adam(learning_rate=CONFIGURATION["LEARNING_RATE"]),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

- **Loss Function:** `categorical_crossentropy` for multi-class classification tasks 
- **Optimizer:** Adam optimizer with a custom learning rate 

---

###  *  **Train the Fine-Tuned Model**

We train the model on the task-specific dataset 

```python
history = fine_tuned_model fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=CONFIGURATION["EPOCHS"],
    callbacks=[early_stopping, lr_scheduler]
)
```

- **Training Dataset:** The dataset specific to the fine-tuning task 
- **Callbacks:** Include learning rate schedulers and early stopping for better convergence 

---

###  *  **Evaluate the Model**

Finally, evaluate the model on the validation or test set 

```python
results = fine_tuned_model evaluate(validation_dataset)
print(f"Validation Accuracy: {results[*]}")
```

- **Results:** The validation accuracy shows the model's performance on unseen data 

---

### Benefits of Fine-Tuning

*  **Leverages Pre-Trained Knowledge:** Fine-tuning allows us to utilize the feature extraction capabilities of a pre-trained model 
 *  **Saves Time and Resources:** Reduces the need for training from scratch on large datasets 
 *  **Adaptability:** The pre-trained model can be adapted to various tasks with minimal modifications 

---

This section demonstrates how to fine-tune a Vision Transformer (ViT) model for specific tasks using TensorFlow and Keras, ensuring efficient and effective utilization of pre-trained models 

---

# ONNX Model Export and Deployment Explanation

## ** *  What is ONNX?**
The Open Neural Network Exchange (ONNX) is an open standard format for machine learning models  It enables seamless interoperability across multiple frameworks, including TensorFlow, PyTorch, and others 

### **Why Use ONNX?**
- **Framework Independence:** Train in one framework and deploy in another 
- **Hardware Acceleration:** Supports GPUs, TPUs, and specialized AI accelerators 
- **Deployment Flexibility:** Enables cloud, mobile, and edge deployment 

---

## ** *  ONNX Export Process**

### **Step *: Prepare the Model**
- Load the trained ViT model in TensorFlow or PyTorch 

### **Step  *: Convert to ONNX**
- **For TensorFlow Models:** Use the `tf *onnx` conversion tool:
  ```bash
  python -m tf *onnx convert --saved-model  /saved_model --output model onnx



#   Now Onnix file used in fast Api

The Emotion Detection API uses a Vision Transformer (ViT) model in ONNX format  This project offers a REST API built using FastAPI for real-time emotion detection from uploaded images 

---

#    *: Deployment Overview

### ** * * Key Components**
- **Model Format:** ONNX
- **Frameworks:** FastAPI, ONNX Runtime
- **Environment:** Localhost, Docker, or cloud services 

### ** *  * Tools Used**
- **API Framework:** FastAPI
- **Model Inference:** ONNX Runtime
- **Deployment Server:** Uvicorn for local development and testing 

---

#    *: Project Structure Breakdown

```
emotion_detection/
  ├── service/
  │   ├── main py            # API entry point
  │   ├── vit_onnx onnx      # ONNX model file
  │   ├── api/
  │   │   ├── api py         # API router setup
  │   │   └── endpoints/
  │   │       └── detect py  # API prediction routes
  │   ├── core/
  │   │   ├── logic/
  │   │   │   └── onnx_infrence py  # Inference logic
  │   │   └── schemas/      # Data models and validation
  └── locusts py            # Load testing script
```

---

#    *: Vision Transformer (ViT) Model Details

## ** * * Model Overview**
The Vision Transformer (ViT) model adapts the Transformer architecture from NLP for image classification  Its use of ONNX format ensures compatibility and fast inference 

### **Model Workflow:**
*  **Image Patching:** Images are split into fixed-size patches 
 *  **Embedding & Encoding:** Patches are embedded and encoded 
 *  **Model Processing:** The Transformer processes the embeddings 
 *  **Classification Head:** Outputs an emotion prediction 

---

#    *: Model Inference Logic (ONNX)

### ** * * File:** `onnx_infrence py`
- **Why:** Loads the ONNX model and performs predictions 
- **How It Works:**
  - **Model Loading:** The ONNX model is loaded during API initialization 
  - **Preprocessing:** Uploaded images are resized and normalized 
  - **Inference Execution:** Runs inference and decodes predictions 

```python
import onnxruntime as rt
import numpy as np
import cv *

def emotion_detector(img):
    if len(img shape) ==  *:
        img = cv * cvtColor(img, cv * COLOR_GRAY *RGB)

    img = cv * resize(img, ( * * *,  * * *))
    img = np expand_dims(np float * *(img), axis= *)

    input_name = s m get_inputs()[ *] name
    output_name = s m get_outputs()[ *] name
    preds = s m run([output_name], {input_name: img})[ *]

    return preds
```

---

#    *: API Endpoints and Routes

### ** * * File:** `detect py`
- **Endpoint:** `/detect`
- **Method:** POST
- **Request:** Upload an image file (JPEG/PNG) 
- **Response:** Predicted emotion with probability 

#### Route Logic Breakdown:
- **Step *:** Accepts file uploads using FastAPI 
- **Step  *:** Calls `emotion_detector()` with the uploaded image 
- **Step  *:** Returns the prediction result 

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO
import numpy as np
from service core logic onnx_infrence import emotion_detector

emotion_router = APIRouter()

@emotion_router post("/detect")
async def detect_emotion(im: UploadFile):
    if im filename split(' ')[-*] not in ['jpg', 'jpeg', 'png']:
        raise HTTPException(status_code= ** *, detail='Unsupported Media Type')

    img = Image open(BytesIO(await im read()))
    img = np array(img)
    prediction = emotion_detector(img)
    
    return {"emotion": prediction}
```

---

#    *: Running the Application

### ** * * Local Deployment**
*  **Install Requirements:**
   ```bash
   pip install -r requirements txt
   ```

 *  **Run the API Server:**
   ```bash
   uvicorn service main:app --host  *  *  *  * --port  * * * *
   ```

 *  **Test Endpoint:**
   Use Postman or cURL:
   ```bash
   curl -X POST "http://localhost: * * * */detect" -F "file=@test_image jpg"
   ```

---
# How to run 
- Run the application using  this `uvicorn service main:app --reload` Command 
- ![image](https://github.com/user-attachments/assets/b1500580-73b8-467e-90d6-522855521cad)
- write `/docs` after the url
- open up this page
- ![image](https://github.com/user-attachments/assets/0129e8f1-b360-41b4-b0bf-f68f913acc00)
-click on the post ` Up button `
- ![image](https://github.com/user-attachments/assets/1df68cd2-b639-4f0b-a4a5-54918b917ae1)
- Click on `Try button ` on right 
- Here input image
- ![image](https://github.com/user-attachments/assets/d61ecb20-340c-4b79-ade7-4e80c7f7ed59)
- Click on exceute
- Hurray !! 🥳 got the result 
- ![image](https://github.com/user-attachments/assets/7519021c-4798-47db-9988-72ae6c9bb320)
- On image and It is actullay labelled as Sad in Dataset
- ![image](https://github.com/user-attachments/assets/262370bb-be2f-4c29-a09c-a18f258f0f27)
 
---

# Conclusion

The Emotion Detection API project showcases the integration of cutting-edge machine learning models with modern API frameworks  By leveraging the Vision Transformer (ViT) model in ONNX format, the API delivers high-performance emotion detection while ensuring compatibility and scalability  

This project is designed with flexibility in mind—whether deployed locally for testing or scaled up using cloud platforms, it maintains efficiency and ease of use  From model inference to API development, every component has been carefully crafted to enable real-time emotion analysis 

---

# Next Steps & Future Improvements

To further enhance this project, consider the following improvements:

- **Model Optimization:** Use techniques like model quantization and ONNX Runtime optimization for faster inference 
- **Deployment Enhancements:** Implement CI/CD pipelines for automated deployments 
- **Scalability:** Expand deployment using container orchestration platforms like Kubernetes 
- **Feature Expansion:** Add emotion visualization tools and real-time monitoring dashboards 
- **Security:** Include authentication, logging, and data privacy compliance for production environments 

---

# Final Thoughts

We hope this guide provides all the necessary details for understanding, running, and deploying the Emotion Detection API  With its modular design, powerful inference capabilities, and cloud-ready architecture, this project is well-suited for real-world applications, research, and continuous development 

Thank you for exploring the Emotion Detection API project  Feel free to contribute, raise issues, or suggest enhancements to improve the system further 
