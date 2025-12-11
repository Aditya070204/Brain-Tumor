# system-libraries
import os
import shutil
import time
import pathlib
import itertools

# data-handling 
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

# deep-learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.applications import ResNet50, EfficientNetB5, VGG16
from keras.optimizers import Adam, Adamax
from keras.losses import CategoricalFocalCrossentropy
from keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.regularizers import l1,l2
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras import regularizers

# ignore-warnings
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# data-path with labels generating
def define_paths(data_dir):
    filepath = []
    label = []

    folds = os.listdir(data_dir)
    for fold in folds:
        files = f"{data_dir}/{fold}"
        # files = os.path.join(data_dir,fold)
        imgs = os.listdir(files)
        for img in imgs:
            path = f"{files}/{img}"
            filepath.append(path)
            if fold=='meningioma':
                label.append('meningioma')
            elif fold=='glioma':
                label.append('glioma')
            elif fold=='pituitary':
                label.append('pituitary tumor')
            elif fold=='notumor':
                label.append('no tumor')
    return filepath, label

# create dataframe
def define_df(files, classes):
    Fseries = pd.Series(files, name='filepaths')
    Cseries = pd.Series(classes, name='labels')

    return pd.concat([Fseries,Cseries], axis=1)

def create_df(data_dir):
    filepath, label = define_paths(data_dir)
    df = define_df(filepath, label)

    train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, stratify=df['labels'], random_state=123)

    valid_df, test_df = train_test_split(dummy_df, test_size=0.5, shuffle=True, random_state=123, stratify=dummy_df['labels'])

    return train_df, valid_df, test_df

# Converting image to tensors via image_data_generator
def create_gens(train_df, valid_df, test_df, batch_size):

    # model parameters
    image_size = (320,320)
    channel = 3
    color = 'rgb'
    image_shape = (image_size[0], image_size[1], channel)
    
    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length//n for n in range(1,ts_length+1) if ts_length%n==0 and ts_length/n<=80]))
    test_steps = ts_length//test_batch_size

    def scalar(img):
        return img
    
    tr_gen = ImageDataGenerator(preprocessing_function=scalar,
                                rescale=1./255,
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest')
    ts_gen = ImageDataGenerator(preprocessing_function=scalar, rescale=1./255)

    train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=image_size, class_mode='categorical', color_mode=color, shuffle=True, batch_size=batch_size)

    valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=image_size, color_mode=color, class_mode='categorical', shuffle=True, batch_size=batch_size)

    test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=image_size, color_mode=color, class_mode='categorical', shuffle=True, batch_size=test_batch_size)

    return train_gen, valid_gen, test_gen

# show sample images
def show_img(gen):
    g_dict = gen.class_indices
    classes = list(g_dict.keys())
    images, labels = next(gen)

    print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")

    length = len(labels)
    sample = min(length,25)
    
    plt.figure(figsize=(20,20))

    for i in range(sample):
        plt.subplot(5,5,i+1)
        image = images[i]
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')  
    plt.show()

# plot value counts for a column in a dataframe
def plot_labels(lcount, labels, values, plot_title):
    width = lcount*4
    width = np.min([width,20])

    plt.figure(figsize=(width,5))

    form = {'family':'serif', 'color':'blue', 'size':25}
    sns.barplot(x=labels, y=values)
    plt.title(f'Image per label in {plot_title} data', fontsize=24, color='blue')
    plt.xticks(rotation=90, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('CLASS', fontdict=form)
    yaxis_label = 'IMAGE COUNT'
    plt.ylabel(yaxis_label, fontdict=form)

    rotation = 'vertical' if lcount>=8 else 'horizontal'
    for i in range(lcount):
        plt.text(i, values[i]/2, str(values[i]), fontsize=12, rotation=rotation, color='yellow', ha='center')
    plt.show()

def plot_label_count(df, plot_title):

    v_counts = df['labels'].value_counts()
    labels = v_counts.keys().to_list()
    values = v_counts.to_list()
    l_count = len(labels)

    if l_count>55:
        print(f"The number of labels is greater than 55, no plot will be produced.")
    else:
        plot_labels(l_count, labels, values, plot_title)


data_dir = "./brain-tumor/Training"
try:
    train_df, valid_df, test_df = create_df(data_dir)
    batch_size = 50
    train_gen, valid_gen, test_gen = create_gens(train_df, valid_df, test_df, batch_size)
    print(train_gen, valid_gen, test_gen)

except:
    print('Invalid Input')

show_img(train_gen)

# Display Class Imbalance
class_counts = train_gen.classes
class_labels = list(train_gen.class_indices.keys())
print(f"Class Distribution: {np.bincount(class_counts)}")

plot_label_count(train_df,"Train")

# Compute Class Weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_counts),
    y=class_counts
)
class_weights = dict(enumerate(class_weights))
print(f"Computed Class Weights: {class_weights}")

# show sample images
def show_img(gen):
    g_dict = gen.class_indices
    classes = list(g_dict.keys())
    images, labels = next(gen)

    print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")

    length = len(labels)
    sample = min(length,25)
    
    plt.figure(figsize=(20,20))

    for i in range(sample):
        plt.subplot(5,5,i+1)
        image = images[i]
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')  
    plt.show()

class MyCallBack(keras.callbacks.Callback):
    def __init__(self, patience, stop_patience, threshold, factor, batches, epochs, ask_epoch):
        super(MyCallBack,self).__init__()
        # self.model = model
        self.patience = patience # specifies how many epochs without improvement before learning rate is adjusted
        self.stop_patience = stop_patience # specifies how many times to adjust lr without improvement to stop training
        self.epochs = epochs
        self.threshold = threshold # specifies training accuracy threshold when lr will be adjusted based on validation loss
        self.factor = float(factor) # factor by which to reduce the learning rate
        print(self.factor,type(self.factor))
        self.batches = batches # number of training batch to run per epoch
        self.ask_epoch = ask_epoch
        self.ask_epoch_initial = ask_epoch # save this value to restore if restarting training

        # callback parameters
        self.count = 0 # how many times lr has been reduced without improvement
        self.stop_count = 0
        self.best_epoch = 0 # epoch with the lowest loss
        self.initial_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate)) # get the initial learning rate and save it
        self.highest_tracc = 0.0 # set highest training accuracy to 0 initially
        self.lowest_vloss = np.inf # set lowest validation loss to infinity initially
        self.best_weights = None # set best weights to model's initial weights
        self.initial_weights = None # save initial weights if they have to get restored
    
    # run when training begins
    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()
        self.initial_weights = self.model.get_weights()
        msg = f"do you want the model to halt training [y/n]? "
        print(msg)
        ans = input('')

        if ans in ['Y','y']:
            self.ask_permission = 1
        else:
            self.ask_permission = 0
        
        msg = "{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:^10s}{9:^8s}".format('Epoch','Loss','Accuracy','V_loss','V_acc','LR','Next LR','Monitor','% Improv','Duration')
        print(msg)
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        stop_time = time.time()
        tr_duration = stop_time - self.start_time
        hours = tr_duration//3600
        minutes = (tr_duration-(hours*3600))//60
        seconds = tr_duration-(hours*3600)-(minutes*60)

        msg = f"Training time elapsed was {int(np.floor(hours))} hrs::{int(np.floor(minutes))} mins::{int(np.floor(seconds))} sec"
        print(msg)
        self.model.set_weights(self.best_weights)

    def on_train_batch_end(self, batch, logs=None):
        acc = logs.get('accuracy')*100
        loss = logs.get('loss')

        msg = '{0:20s}processing batch {1:} of {2:5s}-   accuracy=  {3:5.3f}   -   loss:{4:8.5f}'.format(' ',str(batch), str(self.batches),  acc, loss)
        print(msg, "\r", end= '')

    def on_epoch_begin(self, epoch, logs=None):
        self.ep_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        ep_end = time.time()
        duration = ep_end - self.ep_start

        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        current_lr = lr
        acc = logs.get('accuracy')
        v_acc = logs.get('val_accuracy')
        loss = logs.get('loss')
        v_loss = logs.get('val_loss')

        if acc < self.threshold:
            monitor = 'accuracy'
            if epoch == 0:
                pimprov = 0.0
            else:
                pimprov = (acc-self.highest_tracc)*100/self.highest_tracc # improvement in accuracy
            
            if acc > self.highest_tracc:
                self.highest_tracc = acc
                self.best_weights = self.model.get_weights()
                self.count = 0
                self.stop_count = 0

                if v_loss < self.lowest_vloss:
                    self.lowest_vloss = v_loss
                
                self.best_epoch = epoch+1
            
            else:
                if self.count >= self.patience:
                    lr = lr*self.factor if lr>0.000001 else 0.000001
                    self.model.optimizer.learning_rate.assign(lr)
                    self.count = 0
                    self.stop_count += 1

                else:
                    self.count += 1
        
                if v_loss < self.lowest_vloss:
                    self.lowest_vloss = v_loss

        else:
            monitor = 'val_loss'
            if epoch == 0:
                pimprov = 0
            else:
                pimprov = ((self.lowest_vloss - v_loss)*100)/self.lowest_vloss # improvement in validation loss

            if v_loss < self.lowest_vloss:
                self.lowest_vloss = v_loss
                self.best_weights = self.model.get_weights()
                self.count = 0
                self.stop_count = 0

                if acc > self.highest_tracc:
                    self.highest_tracc = acc
                
                self.best_epoch = epoch+1
            
            else:

                if self.count >= self.patience:
                    lr = lr*self.factor if lr>0.000001 else 0.000001
                    self.model.optimizer.learning_rate.assign(lr)
                    self.count = 0
                    self.stop_count += 1
                
                else:
                    self.count += 1
                
                if acc > self.highest_tracc:
                    self.highest_tracc = acc
        
        msg = f"{str(epoch+1):^3s}/{str(self.epochs):4s} {loss:^9.3f}{acc*100:^9.3f}{v_loss:^9.5f}{v_acc*100:^9.3f}{current_lr:^9.5f}{lr:^9.5f}{monitor:^11s}{pimprov:^10.2f}{duration:^8.2f}"
        print(msg)

        if self.stop_count > self.stop_patience-1:
            msg = f"Training has been halted at epoch {epoch+1} after {self.stop_patience} adjustments of learning rate with no improvement"
            print(msg)
            self.model.stop_training = True
        else:
            if self.ask_epoch != None and self.ask_permission != 0:
                if epoch+1 >= self.ask_epoch:
                    msg = f"enter 'H' to halt training or an integer for number of epochs to run then ask again"
                    print(msg)
                    ans = input('')
                    if ans in ['H','h']:
                        msg = f"Training has been halted at epoch {epoch+1} due to user input."
                        print(msg)
                        self.model.stop_training = True
                    
                    else:
                        try:
                            ans = int(ans)
                            self.ask_epoch += ans

                            msg = f"Training will continue until epoch {str(self.ask_epoch)}"
                            print(msg)
                            msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}'.format('Epoch', 'Loss', 'Accuracy', 'V_loss', 'V_acc', 'LR', 'Next LR', 'Monitor', '% Improv', 'Duration')
                            print(msg)
                        except Exception:
                            print('Invalid')

# create model structure
img_size = (320,320)
channels = 3
image_shape = (img_size[0], img_size[1], channels)
class_count = len(list(train_gen.class_indices.keys()))

# # Create EfficientNetB5 Model with Fine-Tuning
# base_model = EfficientNetB5(weights='imagenet', include_top=False, input_shape=image_shape)

# # Define Model
# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.Dense(1024, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(len(class_labels), activation='softmax')
# ])

# # Freeze the initial layers (first 15 layers)
# for layer in base_model.layers[:400]:
#     layer.trainable = False

# # Unfreeze the last few layers for fine-tuning (from layer 15 onwards)
# for layer in base_model.layers[-20:]:
#     layer.trainable = True



from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("./efficient-net-b5-Brain Tumors-93.06.keras")

model.summary()
# Compile the model
# Compile Model
model.compile(optimizer=Adam(learning_rate=0.00001),loss="categorical_crossentropy",metrics=['accuracy'])

# setting callbacks
batch_size = 50
epochs = 50
patience = 1
stop_patience = 40
threshold = 0.96
factor = 0.8
ask_epoch = 10
batches = int(np.ceil(len(train_gen.labels)/batch_size)) # number of training batch to run per epoch

# Callbacks
logs = "./logs"
os.makedirs(logs, exist_ok=True)
tensorboard_callback = TensorBoard(log_dir=logs, histogram_freq=1)
callback = MyCallBack(patience= patience, stop_patience= stop_patience, threshold= threshold,
            factor= factor, batches= batches, epochs= epochs, ask_epoch= ask_epoch )
early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=40, factor=0.8, min_lr=1e-6)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Train Model
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    epochs=50,
    validation_data=test_gen,
    validation_steps=test_gen.samples // test_gen.batch_size,
    class_weight=class_weights,
    callbacks=[callback,early_stopping, lr_scheduler, model_checkpoint, tensorboard_callback]
)

# Visualize Training Results
def plot_results(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

plot_results(history)

# Evaluate Model
valid_gen.reset()
y_pred = model.predict(valid_gen, steps=valid_gen.samples // valid_gen.batch_size + 1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = valid_gen.classes
accuracy = np.mean(y_pred_classes == y_true)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# model evaluation
ts_length = len(test_df)
test_batch_size = max(sorted([ts_length//n for n in range(1,ts_length+1) if ts_length%n==0 and ts_length/n<80]))
test_steps = ts_length//test_batch_size

train_score = model.evaluate(train_gen, steps=test_steps, verbose=1)
valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=1)
test_score = model.evaluate(test_gen, steps=test_steps, verbose=1)

print(f"Train Loss: {train_score[0]}")
print(f"Train Accuracy: {train_score[1]}")
print("--"*20)
print(f"Validation Loss: {valid_score[0]}")
print(f"Validation Accuracy: {valid_score[1]}")
print("--"*20)
print(f"Test Loss: {test_score[0]}")
print(f"Test Accuracy: {test_score[1]}")

# save model
model_name = 'efficient-net-b5'
subject = 'Brain Tumors'
acc = test_score[1]*100
save_path = ''

save_id = str(f'{model_name}-{subject}-{"%.2f" %round(acc,2)}.keras')
model_save_loc = os.path.join(save_path,save_id)
model.save(model_save_loc)
print(f"Model was saved as {model_save_loc}")

# save weights
weight_save_id = str(f'{model_name}-{subject}.weights.h5')
weight_save_loc = os.path.join(save_path,weight_save_id)
model.save_weights(weight_save_loc)
print(f"Weights are saved as {weight_save_loc}")

# Generate CSV files containing classes indicies & image size
class_dict = train_gen.class_indices
img_size = train_gen.image_shape
height = []
width = []
for _ in range(len(class_dict)):
    height.append(img_size[0])
    width.append(img_size[1])

index_series = pd.Series(list(class_dict.values()), name='class_index')
class_series = pd.Series(list(class_dict.keys()), name='class')
height_series = pd.Series(height, name='height')
width_series = pd.Series(width, name='width')
class_df = pd.concat([index_series, class_series, height_series, width_series], axis=1)
csv_name = f"{subject}-class_dict.csv"
csv_save_loc = os.path.join(save_path,csv_name)
class_df.to_csv(csv_save_loc, index=False)
print(f"Class CSV file was saved as {csv_save_loc}")

# Predict Single Image
image_path = "./brain-tumor/Training/pituitary/1742.jpg"
img = load_img(image_path, target_size=(320, 320))

# Display the image
plt.imshow(img)
plt.axis('off')  # Turn off axis
plt.show()

# Preprocess Image
img_array = img_to_array(img)
img_array = np.expand_dims(img_array / 255.0, axis=0)  # Normalize and expand dimensions

# Make Prediction
predictions = model.predict(img_array)
class_index = np.argmax(predictions)
confidence = predictions[0][class_index]

# Class labels and final result
predicted_label = class_labels[class_index]
print(f"Predicted Tumor Class: {predicted_label}, Confidence: {confidence:.2f}")
