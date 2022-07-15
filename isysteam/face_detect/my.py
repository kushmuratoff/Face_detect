from deepface import DeepFace

obj = DeepFace.analyze(img_path = "samar.jpg", actions = ['age', 'gender', ], enforce_detection=False)
print(obj["age"],obj["gender"])


obj = DeepFace.analyze(img_path = "ruzmat.jpg", actions = ['age', 'gender', ], enforce_detection=False)
print(obj["age"],obj["gender"])