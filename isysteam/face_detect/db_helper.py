import sqlite3
import pandas as pd
class DBHelper:
     def __init__(self,db_name):
         try:
             self.conn=sqlite3.connect(db_name,check_same_thread=False)
             self.conn.row_factory=sqlite3.Row
             self.cursor=self.conn.cursor()

         except Exception:
             pass

     def face_detect_person(self):
         try:
             df = pd.read_sql_query('SELECT * FROM face_detect_person;', self.conn)
             return df
         except Exception as ex:
             print("Personda zapsros")

     def face_detect_vaqti(self, PerId):
         try:
             df = pd.read_sql_query('SELECT max(id) FROM face_detect_vaqti where Output_per is null and PesonId_id=' + str(PerId), self.conn)

             # df = pd.DataFrame(df)

             return df
         except Exception as ex:
             print("face_detect_vaqti   ", ex)

     def face_detect_boshi(self):
         try:
             df = pd.read_sql_query('SELECT * FROM face_detect_vaqti ', self.conn)

             # df = pd.DataFrame(df)

             return df
         except Exception as ex:
             print("face_detect_vaqti   ",ex)

     # def facecontrol_davomat_write(self,talaba_id, img_adres,todays_date,stat):
     #     try:
     #         sqlite_insert_with_param = """INSERT INTO facecontrol_davomat
     #                                            (sana, talaba_id_id, rasm , status)
     #                                            VALUES (?, ?, ?, ?);"""
     #         data_tuple = (todays_date, talaba_id, img_adres, stat)
     #         self.cursor.execute(sqlite_insert_with_param, data_tuple)
     #         self.conn.commit()
     #         print('succesful writed')
     #     except sqlite3.Error as error:
     #         pass
#      def deleteSqliteRecord(self,id):
#          try:
#              sql_update_query = """DELETE FROM Keys where generated_id = ?"""
#              self.cursor.execute(sql_update_query, (str(id),))
#              self.conn.commit()
#          except sqlite3.Error as error:
#              pass
#
#      def Chanel(self):
#          try:
#              df = pd.read_sql_query('SELECT * FROM Chanels;', self.conn)
#              return df
#          except Exception:
#              print(Exception)
#              pass
     def Person_write(self):
         try:
             sqlite_insert_with_param = """INSERT INTO face_detect_person
                                                (Ism)
                                                VALUES (?);"""
             data_tuple = ('Person',)
             self.cursor.execute(sqlite_insert_with_param, data_tuple)
             self.conn.commit()
         except sqlite3.Error as error:
             print("Person_write   ",error, "csdcvsd")

     def Vaqt_write(self,Perid,rasm,timeI):
         try:
             sqlite_insert_with_param = """INSERT INTO face_detect_vaqti
                                                (PesonId_id,Input_per,Vaqti_in)
                                                VALUES (?,?,?);"""
             data_tuple = (str(Perid),rasm,timeI)
             self.cursor.execute(sqlite_insert_with_param, data_tuple)
             self.conn.commit()
         except sqlite3.Error as error:
             print("Vaqt_write", error)


     def VaqtAdd(self,Perid,rasm,timeI):
         try:
             sqlite_insert_with_param = """INSERT INTO face_detect_vaqti
                                                (PesonId_id,Input_per,Vaqti_in)
                                                VALUES (?,?,?);"""
             data_tuple = (str(Perid),rasm,timeI)
             self.cursor.execute(sqlite_insert_with_param, data_tuple)
             self.conn.commit()
         except sqlite3.Error as error:
             print("VaqtAdd   ",error,"csdcvsd")

     def VaqtUpdate(self, id, VaqtiI, RasmL):
         try:
             sql = ''' UPDATE face_detect_vaqti
                           SET Input_per = ?, Vaqti_in= ?                               
                           WHERE id = ?'''
             data_tuple = (RasmL, VaqtiI, id)
             # self.conn.cursor()
             self.cursor.execute(sql, data_tuple)

             self.conn.commit()

         except sqlite3.Error as error:
             print("VaqtUpdate ", error)

     def VaqtUpdate_out(self,id,VaqtiI,RasmL):
         try:
             sql = ''' UPDATE face_detect_vaqti
                           SET Output_per = ?, Vaqti_out= ?                               
                           WHERE id = ?'''
             data_tuple = (RasmL,VaqtiI,id)
             # self.conn.cursor()
             self.cursor.execute(sql, data_tuple)

             self.conn.commit()

         except sqlite3.Error as error:
             print("VaqtUpdateout ", error)

     def Person_update(self,id):
         try:
             sql = ''' UPDATE face_detect_person
                           SET Ism = ?                                
                           WHERE id = ?'''
             data_tuple = ('Person'+str(id),id,)
             # self.conn.cursor()
             self.cursor.execute(sql, data_tuple)

             self.conn.commit()

         except sqlite3.Error as error:
             print("Person_update  ", error)
#      def deleteChanel(self,id):
#          try:
#              sql_update_query = """DELETE FROM Chanels where generated_id = ?"""
#              self.cursor.execute(sql_update_query, (str(id),))
#              self.conn.commit()
#          except sqlite3.Error as error:
#              pass
#
# def create_connection(db_file):
#     conn = None
#     try:
#         conn = sqlite3.connect(db_file)
#     except Error as e:
#         print(e)
#     return conn
# def Key(conn):
#     try:
#         df = pd.read_sql_query('SELECT * FROM facecontrol_talabalar;', conn)
#         return df
#     except Exception:
#         print(Exception)
#         pass
#
# def main():
#     database = r"C:\django\Face_Control\db.sqlite3"
#
#     # create a database connection
#     conn = create_connection(database)
#     with conn:
#         # create a new project
#         project = ('Cool App with SQLite & Python', '2015-01-01', '2015-01-30');
#         df = Key(conn)
#         print(df['rasm'],df['id'])
# if __name__ == '__main__':
#     main()