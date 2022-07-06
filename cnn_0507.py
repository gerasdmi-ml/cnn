from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image  import ImageDataGenerator
from tensorflow.keras.utils import img_to_array,load_img
import matplotlib.pyplot as plt
from glob import glob
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

case = 1
#https://github.com/Alexamannn/CNN_from_scratch/blob/main/cnn_Custom_vgg13.ipynb
batch_size_initial = 4


if case==0:
  train_path = "D:/fruits/fruits-360_dataset/fruits-360/Training/"
  test_path = "D:/fruits/fruits-360_dataset/fruits-360/Test/"
elif case==1:
  train_path = "D:/sql_pic/session_116/train/"
  test_path = "D:/sql_pic/session_116/test/"




if case==0:
    img = load_img(train_path + "Apple Braeburn/0_100.jpg")
elif case==1:
    fnames = os.listdir(train_path + "0/")
    img = load_img(train_path + "0/"+fnames[0])
plt.imshow(img)
plt.title("Apple Apple Braeburn")
plt.axis("off")
#plt.show()


shape_of_image = img_to_array(img)
print(shape_of_image.shape)

classes = glob(train_path + "/*")
number_of_class = len(classes)
print("Number of class : " , number_of_class)


## creating flow generator
train_datagen = ImageDataGenerator(rescale = 1./255,
                   shear_range = 0.3,
                   horizontal_flip = True,
                   zoom_range = 0.3)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_path,
                                                   target_size = shape_of_image.shape[:2],
                                                   batch_size = batch_size_initial,
                                                   color_mode = 'rgb',
                                                   class_mode = 'categorical')
test_generator = test_datagen.flow_from_directory(test_path,
                                                   target_size = shape_of_image.shape[:2],
                                                   batch_size = batch_size_initial,
                                                   color_mode = 'rgb',
                                                   class_mode = 'categorical')


x,y = train_generator.next()
fig = plt.figure(figsize =(30,5))
for i in range(2):
    image = x[i]
    plt.imshow(image)
    #plt.show()


# Custom CNN Architeture

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=shape_of_image.shape))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=shape_of_image.shape))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3, 3), activation='relu', input_shape=shape_of_image.shape))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_class, activation='softmax'))
model.compile(loss='categorical_crossentropy',
               optimizer='rmsprop',
               metrics=['accuracy'])
batch_size = batch_size_initial


if case==0:
  number_of_batch = 1600 // batch_size
  number_of_batch_validation = 800
elif case==1:
  number_of_batch = 8000 // batch_size
  number_of_batch_validation = 600

hist = model.fit(
     train_generator,
     steps_per_epoch=number_of_batch,
     epochs=100,
     validation_data=test_generator,
     validation_steps=number_of_batch_validation // batch_size )

# overfitting check

print(hist.history.keys())
plt.plot(hist.history["loss"], label="Train Loss")
plt.plot(hist.history["val_loss"], label="Validaton Loss")
plt.legend()
#plt.show()


plt.figure()
plt.plot(hist.history["accuracy"],label = "Train Accuracy")
plt.plot(hist.history["val_accuracy"],label = "Validaton Accuracy")
plt.legend()
#plt.show()


model.save("CNN_model", save_format="h5")

quit()

#Custom CNN prediction results

x, y = train_generator.next()
y_pred = model.predict(x)

classnamess = list(test_generator.class_indices.keys())
import numpy as np

fig = plt.figure(figsize=(16, 9))
for i, idx in enumerate(np.random.choice(len(test_generator), size=16, replace=False)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x[i]))
    pred_idx = np.argmax(y_pred[i])
    true_idx = np.argmax(y[i])
    ax.set_title("{} ({})".format(classnamess[pred_idx], classnamess[true_idx]),
                  color=("green" if pred_idx == true_idx else "red"))



