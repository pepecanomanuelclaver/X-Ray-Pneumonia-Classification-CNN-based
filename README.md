# CXR_project
This project focuses on detecting the type of pneumonia that is visible in an X-ray message and whether or not COVID-19 was present. 
To assess this two VGG-16 models and a SVM model with Linear binary patterns were compared.

### Dataset
the dataset we used can be downloaded with this line:
```python
!darwin dataset pull v7-labs/covid-19-chest-x-ray-dataset:all-images
```
### Data augmentation
For data augmentaton and saving Keras ImageDataGenerator and flow function are used. 

Augmented using the following parameters:
```python
generator = ImageDataGenerator(
    rotation_range = 10,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    brightness_range = [0.2,1.2])
```
saving is done by using the following settings, where the flow would get broken after the number of augmentations desired per image would break:
```python
generator.flow( image, batch_size=1, save_to_dir = storeloc, shuffle = True, save_prefix = "aug", save_format = "png")
```

### data loading VGG
Dataloading was done using keras flow_from_dataframe with the following parameters:
```python
flow_from_dataframe(
    dataframe = df,
    directory = image_loc,
    x_col = filename,
    y_col = label,
    batch_size = (60 train, 40 val, 1 test),
    seed = 4,
    suffle = True (train,val) False(test),
    target_size = (224,224), #standard VGG size
    keep_aspect_ratio = True,
    validate_filenaes = True)
```

### Model parameters
##### VGG-16 
Function to create the model where trainable was False for the top_layer for both models and trainable was False for the frozen model. Weights was 'imagnet'.

```python
def get_model(weight, include_top_layer, input_size, trainable, classes):
    vgg =  VGG16(weights=weight, include_top=include_top_layer, input_shape = input_size)
    model = Sequential()
    model.add(vgg)
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(classes, activation='softmax'))
    model.layers[0].trainable = trainable
    model.compile(loss='categorical_crossentropy', 
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
                  metrics=["accuracy"])
    return model
```

##### SVM

``` python
#Helper function that creates SVMs. As mentioned below, C=1000 and gamma=1 were previously found using Gridsearch
def create_svm(x_train, y_train):
    model = svm.SVC(C=1000, gamma=1)
    model.fit(x_train, y_train)
    return model
```
