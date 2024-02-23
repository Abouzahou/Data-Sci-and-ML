from imageai.Classification import ImageClassification

import os

execution_path = os.getcwd()  # Get the current working directory

#Setting up the model
prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2() #using MobileNetV2 Algorithm #small size algorithm #moderate accuracy
prediction.setModelPath(os.path.join(execution_path, "mobilenet_v2-b0353104.pth")) #path of model
prediction.loadModel()

#Making the prediction
predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "house.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)