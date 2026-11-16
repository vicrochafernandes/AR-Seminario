from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("WORKSPACE").project("PROJECT")
dataset = project.version(1).download("yolov8")  # creates data.yaml + train/valid/test