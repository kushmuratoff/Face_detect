from datetime import datetime
import numpy as np
import os
from face_detect.db_helper import DBHelper
db = DBHelper("D:\Camera\isysteam\db.sqlite3")
# Current date time in local system
# print((db.face_detect_vaqti()))
#
# maxVid = np.array(db.face_detect_vaqti())
# print(maxVid[0][5])
db.Person_update(23)
my_maxid = np.array(db.face_detect_person()['id']).max()
print(os.path.split(os.getcwd())[0],"cssd")
