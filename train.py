import os
import datetime
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from xml.etree import ElementTree as ET
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from keras.preprocessing.image import save_img
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50  # Import the ResNet50 model
from keras.layers import GlobalAveragePooling2D, Dense
from keras import Model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Paths to dataset, plots and weights
images_path = '/home/nat/Dog/images/'
annotations_path = '/home/nat/Dog/Annotations/'
plot_path = '/home/nat/Dog/plots/SDD_R50Ver01_P01.png'
roc_plot_dir = '/home/nat/Dog/plots/SDD_R50Ver01_P01_ROC'
CheckpointWeights_save_path = '/home/nat/Dog/SDD_R50Ver01.weights.h5'
KerasModel_save_path = '/home/nat/Dog/plots/SDD_R50Ver01_P01.h5'
tensorboard_log_dir = '/home/nat/Dog/logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Function to parse XML annotations
def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    bbox = root.find('object').find('bndbox')
    label = root.find('object').find('name').text
    return {
        'label': label,
        'bbox': [
            int(bbox.find('xmin').text),
            int(bbox.find('ymin').text),
            int(bbox.find('xmax').text),
            int(bbox.find('ymax').text)
        ]
    }

# Load data and annotations
data = []
labels = []
for breed_dir in os.listdir(images_path):
    breed_images_path = os.path.join(images_path, breed_dir)
    breed_annotations_path = os.path.join(annotations_path, breed_dir)
    for image_file in os.listdir(breed_images_path):
        image_path = os.path.join(breed_images_path, image_file)
        annotation_path = os.path.join(breed_annotations_path, os.path.splitext(image_file)[0])
        annotation = parse_annotation(annotation_path)
        data.append(image_path)
        labels.append(annotation['label'])

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode labels
label_to_index = {label: idx for idx, label in enumerate(np.unique(labels))}
index_to_label = {idx: label for label, idx in label_to_index.items()}
labels = np.array([label_to_index[label] for label in labels])

# Specify random seed, using KFold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

for train_index, test_index in kf.split(data, labels):
    train_data, test_data = data[train_index], data[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

datagen = ImageDataGenerator(
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # brightness_range=[0.8, 1.2],
    # fill_mode='nearest'
)

# Load and preprocess images
def load_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))  # Adjusted for ResNet50
    image = tf.keras.preprocessing.image.img_to_array(image)
    return image

train_images = np.array([load_image(image_path) for image_path in train_data])
test_images = np.array([load_image(image_path) for image_path in test_data])

train_images = train_images / 255.0
test_images = test_images / 255.0

print("Number of train images:", len(train_images))
print("Number of test images:", len(test_images))

train_save_path = '/home/nat/Dog/Processed_Train_Images/'
test_save_path = '/home/nat/Dog/Processed_Test_Images/'

# Create directories if they don't exist
os.makedirs(train_save_path, exist_ok=True)
os.makedirs(test_save_path, exist_ok=True)

# Save the train images
for i, img in enumerate(train_images):
    save_img(os.path.join(train_save_path, f"train_image_{i}.png"), img)

# Save the test images
for i, img in enumerate(test_images):
    save_img(os.path.join(test_save_path, f"test_image_{i}.png"), img)

print("Train and test images have been saved.")

#Build
#Case 1: ResNet50
lr = 0.01
batch_size = 32
epoch = 50
def build_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Load ResNet50 with ImageNet weights
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # Output layer for your classes
    model = Model(inputs=base_model.input, outputs=predictions)
    #Compile the model
    adam = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=adam,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
    return model

model = build_model(num_classes=120)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=KerasModel_save_path,
        save_weights_only=False,
        save_best_only=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=CheckpointWeights_save_path,
        save_weights_only=True,
        save_best_only=True
    ),
    tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1),
]

# Set TensorFlow logger level to INFO
tf.get_logger().setLevel('INFO')  # Place it here

#Train
train_generator = datagen.flow(train_images, train_labels, batch_size=batch_size)

history = model.fit(
    train_generator,
    validation_data=(test_images, test_labels),
    epochs=epoch,
    callbacks=callbacks
)

#Result
#Accuracy and Loss
model.summary()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(plot_path)

#ROC Curves are split into 10 classes
# Create directory for ROC plots if not exists
if not os.path.exists(roc_plot_dir):
    os.makedirs(roc_plot_dir)

# Compute ROC curve and ROC area for each class
n_classes = len(label_to_index)
test_labels_binarized = label_binarize(test_labels, classes=range(n_classes))
y_score = model.predict(test_images)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_labels_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves, 12 classes per plot
for i in range(0, n_classes, 12):
    plt.figure()
    for j in range(i, min(i + 12, n_classes)):
        plt.plot(fpr[j], tpr[j], label=f'Class {index_to_label[j]} (area = {roc_auc[j]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - Classes {i+1} to {min(i+12, n_classes)}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(roc_plot_dir, f'ROC_curve_{i//12 + 1}.png'))
    plt.close()
