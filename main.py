
import tensorflow as tf
import cv2
import glob
from scipy import ndimage
import random
import numpy as np
from sklearn.model_selection import train_test_split


def read_images(path):
    images = []
    for img in glob.glob(path):
        images.append(cv2.imread(img))
    return images


#This function is responsible for copying images from keys folder and pasting them randomly (random angle). Knowing where to paste wrench object, we also know
#bounding box coordinates. They are crucial for teaching CNN. 'dataset' folder can be empty as our data is generated from images located in 'background' and 'keys'
#folders
def image_dataset():
    iter = 0
    keys = read_images('./keys/*.jpg')
    background = read_images('./background/*.jpg')
    keys_resized = [cv2.resize(img, (0,0), fx=0.25, fy=0.25) for img in keys]
    backs_resized = [cv2.resize(img, (224,224)) for img in background]
    bboxes = []
    filenames = []

    print(len(keys))
    for back in backs_resized:
        for i in range(20):
            tool = keys_resized[random.randint(0, 5)]
            tool = ndimage.rotate(tool, random.randint(0,360))
            back_copy = back.copy()
            height_tool, width_tool,_ = tool.shape
            height_back, width_back, _ = back.shape
            y = random.randint(0, height_back - height_tool)
            x = random.randint(0, width_back - width_tool)
            area = back[y:y+height_tool, x:x+width_tool]
            gray_tool = cv2.cvtColor(tool, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_tool, 11, 255, cv2.THRESH_BINARY)
            inv = cv2.bitwise_not(mask)
            back_blackout = cv2.bitwise_and(area,area,mask=inv)
            take_tool = cv2.bitwise_and(tool,tool,mask = mask)
            back[y:y+height_tool,x:x+width_tool] = cv2.add(back_blackout,take_tool)
            bboxes.append((float(x)/width_back,float(y)/height_back,float(x+width_tool)/width_back, float(y+height_tool)/height_back))
            filenames.append(str(iter)+'.jpg')
            cv2.imwrite('dataset/'+str(iter)+'.jpg', back)
            iter = iter+1
            back = back_copy.copy()

    data = read_images('./dataset/*.jpg')
    data_ready = [cv2.resize(img, (224,224)) for img in data]
    data_ready = np.array(data_ready, dtype="float32")/255.0
    filenames = np.array(filenames)
    bboxes = np.array(bboxes, dtype ="float32")
    return data_ready, filenames, bboxes


data, filenames, bboxes = image_dataset()


split = train_test_split(data,filenames,bboxes, test_size=0.10, random_state=42)
(trainImages, testImages) = split[:2]
(trainFilenames, testFilenames) = split[2:4]
(trainbboxes, testbboxes) = split[4:]

print(len(trainImages))

model_vgg = tf.keras.applications.VGG16(weights ="imagenet", include_top = False, input_tensor = tf.keras.layers.Input(shape=(224,224,3)))

model_vgg.trainable = False

flatten = model_vgg.output
flatten = tf.keras.layers.Flatten() (flatten)

bboxHead = tf.keras.layers.Dense(128, activation ="relu") (flatten)
bboxHead = tf.keras.layers.Dense(64, activation ="relu") (bboxHead)
bboxHead = tf.keras.layers.Dense(32, activation ="relu") (bboxHead)
bboxHead = tf.keras.layers.Dense(4, activation ="sigmoid") (bboxHead)

#There's how it works: We give an image on input - > receive bounding box coordinates on output.
model = tf.keras.models.Model(inputs = model_vgg.input, outputs = bboxHead)

opt = tf.keras.optimizers.Adam(learning_rate = 0.0001)
model.compile(loss="mse", optimizer =opt, metrics = ['accuracy'])

H = model.fit(
    trainImages, trainbboxes,
    validation_data =(testImages, testbboxes),
    batch_size=2,
    epochs=10,
    verbose=1
)

#Possible saving/loading model
#model.save('model.h5')
#model = tf.keras.models.load_model('model.h5')


test_set_img = read_images('./test_set/*.jpg')

#Testing our neural net on downloaded images. With 10 epochs it's quasi-efective. Possible improvements - more epochs.
for image in test_set_img:
    ig = image
    image = cv2.resize(image, (224,224))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)[0]
    (startX, startY, endX, endY) = preds
    ig=cv2.resize(ig, (224,224))
    (h,w) = ig.shape[:2]

    startX = int(startX*w)
    startY = int(startY*h)
    endX = int(endX*w)
    endY = int(endY*h)

    cv2.rectangle(ig, (startX, startY), (endX, endY), (0,255,0), 2)
    cv2.imshow("predict_rect", ig)
    cv2.waitKey(0)


