import numpy as np
from imutils.video import VideoStream
import cv2, os
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
from db_helper import DBHelper
db = DBHelper(os.path.split(os.getcwd())[0] + "/db.sqlite3")

prototxtPath=os.path.sep.join([r'model','deploy.prototxt'])
weightsPath=os.path.sep.join([r'model','res10_300x300_ssd_iter_140000.caffemodel'])

faceNet=cv2.dnn.readNet(prototxtPath,weightsPath)

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def embedding_vec():
    try:
        dataset = datasets.ImageFolder(os.path.split(os.getcwd())[0] + '/media/images')
        idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        def collate_fn(x):
            return x[0]
        loader = DataLoader(dataset, collate_fn=collate_fn)
        name_l = []
        embedding_l = []

        for img, idx in loader:
            face, prob = mtcnn(img, return_prob=True)
            if face is not None and prob > 0.90:  # if face detected and porbability > 90%
                emb = resnet(face.unsqueeze(0))  # passing cropped face into resnet model to get embedding matrix
                embedding_l.append(emb.detach())  # resulten embedding matrix is stored in a list
                name_l.append('/media/images/'+str(idx_to_class[idx]))  # names are stored in a list
        return embedding_l, name_l
    except Exception as ex:
        pass

embedding_list,name_list = embedding_vec()

def face_detect(img_path, face_m):
    try:
        img = Image.open(img_path)
        face, prob = mtcnn(img, return_prob=True)
        emb = resnet(face.unsqueeze(0)).detach()
        dist_list = []  # list of matched distances, minimum distance is used to identify the person
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)

        idx_min = dist_list.index(min(dist_list))

        if min(dist_list) < 0.6:
            maxVid = db.face_detect_vaqti(int(name_list[idx_min][20:])).iloc[0, 0]
            if (maxVid != None):
                cv2.imwrite(str(os.path.split(os.getcwd())[0])+name_list[idx_min] + "/" + str(int(maxVid)) + "_out.jpg", face_m)
                db.VaqtUpdate_out(int(maxVid), datetime.now(), name_list[idx_min] + "/" + str(int(maxVid)) + "_out.jpg")
    except Exception as ex:
        pass
    return 0

def output_capture():
    try:
        vs = VideoStream(src=0).start()
        while True:
            frame = vs.read()
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            faceNet.setInput(blob)
            detections = faceNet.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype('int')
                    # ensure the bounding boxes fall within the dimensions of the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    face_m = frame[startY:endY, startX:endX]
                    color = (0, 0, 255)
                    if endY - startY > 290 and endY - startY < 350 and endX - startX > 190 and endX - startX < 300:
                        color = (0, 255, 0)
                        cv2.imwrite("frame.jpg", face_m)
                        face_detect("frame.jpg", face_m)

                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        vs.release()
        vs.stop()
    except Exception as ex:
        pass

output_capture()