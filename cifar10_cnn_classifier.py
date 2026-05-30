import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,recall_score,f1_score,classification_report
)
# LOAD DATASET
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train_original = y_train.copy()
y_test_original = y_test.copy()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#LOSS FUNCTION
def custom_loss(y_true, y_pred):
    ce_loss = tf.keras.losses.categorical_crossentropy(
        y_true,y_pred
    )
    confidence_penalty = tf.reduce_mean(
        tf.square(y_pred)
    )
    total_loss = ce_loss + 0.1 * confidence_penalty
    return total_loss
# CNN MODEL
model = models.Sequential()
model.add(
    layers.Conv2D(
        32,(3, 3),activation='relu',padding='same',input_shape=(32, 32, 3))
)
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(
    layers.Conv2D(
        64,(3, 3),activation='relu',padding='same'
    )
)
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(
    layers.Conv2D(
        128,(3, 3),activation='relu',padding='same'
    )
)
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())

model.add(
    layers.Dense(
        256,activation='relu'
    )
)
model.add(layers.Dropout(0.5))

model.add(
    layers.Dense(
        10,activation='softmax'
    )
)

# COMPILE MODEL

model.compile(
    optimizer='adam',loss=custom_loss,metrics=['accuracy']
)
# MODEL SUMMARY
model.summary()
# TRAIN MODEL
history = model.fit(
    x_train,y_train,epochs=25,batch_size=64,validation_data=(x_test, y_test)
)
# EVALUATE MODEL
loss, accuracy = model.evaluate(
    x_test,y_test,verbose=1
)
print("\nMODEL PERFORMANCE")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy*100:.2f}%")

predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)
y_true = y_test_original.flatten()

precision = precision_score(
    y_true,y_pred,average='weighted'
)
recall = recall_score(
    y_true,y_pred,average='weighted'
)
f1 = f1_score(
    y_true,y_pred,average='weighted'
)
print("ADDITIONAL METRICS\n")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nCLASSIFICATION REPORT\n")

print(
    classification_report(
        y_true,y_pred,target_names=class_names
    )
)

print("\nSAMPLE PREDICTIONS\n")

for i in range(10):

    actual_label = class_names[y_true[i]]
    predicted_label = class_names[y_pred[i]]

    print(f"\nImage {i+1}")
    print(f"Actual Label    : {actual_label}")
    print(f"Predicted Label : {predicted_label}")


for i in range(5):

    plt.figure(figsize=(3,3))
    plt.imshow(x_test[i])
    plt.title(
        f"Actual: {class_names[y_true[i]]}\n"
        f"Predicted: {class_names[y_pred[i]]}"
    )
    plt.axis("off")
    plt.show()
#graph
plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)
plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
