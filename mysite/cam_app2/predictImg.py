import cv2
from cam_app import predictModel
import numpy as np
from PIL import Image
from io import BytesIO
labelNames = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
def preditImg(path):
    if predictModel.model == None:
        predictModel.load_model()
    # 从文件读取图片
    img = cv2.imread("mysite"+path)
    # 转为灰度图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用cascade进行人脸检测 faceRects为返回的结果
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(120, 120))

    # 使用enumerate 函数遍历序列中的元素以及它们的下标
    # 下标i即为人脸序号
    for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            face_arr = face.astype(np.float32)
            face_arr /= 255.
            face_arr = np.expand_dims(face_arr, axis=0)
            predictions = predictModel.model.predict(face_arr)

            center = (x + w // 2, y + h // 2)
            img = cv2.ellipse(img, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
            i = np.argmax(predictions, axis=1)
            cv2.putText(img, labelNames[i[0]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, 8)
    img = img[:, :, ::-1]
    ndarray_convert_img = Image.fromarray(img,mode ='RGB')
    img_byte = BytesIO()
    ndarray_convert_img.save(img_byte, format='JPEG')

    return img_byte.getvalue()