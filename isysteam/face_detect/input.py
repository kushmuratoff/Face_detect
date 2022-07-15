import numpy as np
from imutils.video import VideoStream
import cv2, os
from time import gmtime, strftime
from imutils import paths
from datetime import datetime

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
# from isysteam.face_detect.db_helper import DBHelper
from db_helper import DBHelper

db = DBHelper(os.path.split(os.getcwd())[0] + "/db.sqlite3")
print(os.path.split(os.getcwd())[0])
prototxtPath=os.path.sep.join([r'model','deploy.prototxt'])
weightsPath=os.path.sep.join([r'model','res10_300x300_ssd_iter_140000.caffemodel'])

faceNet=cv2.dnn.readNet(prototxtPath,weightsPath)


mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

name_list = []
embedding_list = []

def embedding_vec():
    try:
        dataset = datasets.ImageFolder(str(os.path.split(os.getcwd())[0]) + '/media/images')  # photos folder path
        idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}  # accessing names of peoples from folder names

        def collate_fn(x):
            return x[0]

        loader = DataLoader(dataset, collate_fn=collate_fn)

        name_l = []  # list of names corrospoing to cropped photos
        embedding_l = []  # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

        for img, idx in loader:
            face, prob = mtcnn(img, return_prob=True)
            if face is not None and prob > 0.90:  # if face detected and porbability > 90%
                emb = resnet(face.unsqueeze(0))  # passing cropped face into resnet model to get embedding matrix
                embedding_l.append(emb.detach())  # resulten embedding matrix is stored in a list
                name_l.append('/media/images/'+str(idx_to_class[idx]))  # names are stored in a list
        return embedding_l, name_l
    except Exception as ex:
        pass


# bazada rasmi umuman rasmi yo'q, control oynaga paralel qo'shildi va yangi papka hosil qilindi
def TimeCaseOne(emb,face_m):
    try:
        db.Person_write()
        my_maxid = np.array(db.face_detect_person()['id']).max()
        path = "/media/images/Person" + str(my_maxid)
        db.Person_update(int(my_maxid))
        os.makedirs(str(os.path.split(os.getcwd())[0]) + path)
        embedding_list.append(emb.detach())
        name_list.append(path)
        link = str(strftime("%m_%d_%Y_%H_%M_%S_", gmtime())) + ".jpg"
        cv2.imwrite(str(os.path.split(os.getcwd())[0])+path + "/" + link, face_m)
        maxVid = db.face_detect_vaqti(my_maxid).iloc[0, 0]
        rasm_in = path + "/" + str(maxVid)+ "_in.jpg"
        db.Vaqt_write(my_maxid, rasm_in, datetime.now())
    except Exception as ex:
        pass



# oldin bazada bor va kirib chiqqan va yangi holatni yozish
def TimeCaseTwo(PerId,rasmL,face_m):
    try:

        if(db.face_detect_boshi().count()["id"]==0):
            db.VaqtAdd(PerId, rasmL, datetime.now())
        else:
            maxVid = db.face_detect_vaqti(PerId).iloc[0,0]
            if (maxVid != None):
                # myId = np.array(db.face_detect_vaqti(PerId)['id']).max()
                # print(rasmL+str(int(maxVid))+".jpg")
                cv2.imwrite(str(os.path.split(os.getcwd())[0]) + rasmL+str(int(maxVid))+"_in.jpg", face_m)
                db.VaqtUpdate(int(maxVid), datetime.now(), rasmL+str(int(maxVid))+"_in.jpg")
                # updete last item
                # id , rasm link,
            else:
                # insert next item
                # person id vaqti rasm linki kerak
                cv2.imwrite(str(os.path.split(os.getcwd())[0]) + rasmL+str(int(PerId))+"_in.jpg", face_m)
                db.VaqtAdd(PerId,rasmL+str(int(PerId))+"_in.jpg",datetime.now())
    except Exception as ex:
        pass




def face_match(img_path,face_m):
    try:
        img = Image.open(img_path)
        face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
        emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false

        dist_list = [] # list of matched distances, minimum distance is used to identify the person
        if len(embedding_list)>0:
            for idx, emb_db in enumerate(embedding_list):
                dist = torch.dist(emb, emb_db).item()
                dist_list.append(dist)
            idx_min = dist_list.index(min(dist_list))
            if min(dist_list)>0.6:
                #will go case one
                TimeCaseOne(emb,face_m)

            else:
                path_img = list(paths.list_images(str(os.path.split(os.getcwd())[0]) + str(name_list[idx_min])))
                adres=str(name_list[idx_min])
                link1 = str(strftime("%m_%d_%Y_%H_%M_%S", gmtime())+".jpg")
                link = str(name_list[idx_min])+"/"
                TimeCaseTwo(int(adres[20:]),link,face_m)

                if len(path_img)<10:
                    embedding_list.append(emb.detach())
                    name_list.append(name_list[idx_min])
                    cv2.imwrite(str(os.path.split(os.getcwd())[0]) + link+link1, face_m)

        else:
            TimeCaseOne(emb,face_m)

    except Exception as ex:
        pass
    return 0

def capture():
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
                    # we need the X,Y coordinates
                    #             try:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype('int')
                    # ensure the bounding boxes fall within the dimensions of the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    # extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
                    face_m = frame[startY:endY, startX:endX]
                    color = (0, 0, 255)
                    if endY - startY > 290 and endY - startY < 350 and endX - startX > 190 and endX - startX < 300:
                        # print(startY - endY, "  ", startX - endX)
                        cv2.imwrite("frame.jpg", face_m)
                        color = (0, 255, 0)
                        face_match("frame.jpg", face_m)
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
if np.array(db.face_detect_person()['id'].count())>0:
    embedding_list,name_list = embedding_vec()

capture()
