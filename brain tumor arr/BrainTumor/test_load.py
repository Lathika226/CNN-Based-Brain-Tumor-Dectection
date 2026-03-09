import os
# os.environ["TF_USE_LEGACY_KERAS"] = "1"
from tensorflow.keras.models import model_from_json, Model # type: ignore
from tensorflow.keras.utils import custom_object_scope # type: ignore
t = open("Model/segmented_model.json").read()
with custom_object_scope({'Model': Model}):
    m = model_from_json(t)
print("Loaded functional model successfully")

try:
    with open('Model/model.json', "r") as json_file:
           loaded_model_json = json_file.read()
           classifier = model_from_json(loaded_model_json)
           print("Loaded Sequential model successfully")
except Exception as e:
    print("Sequential model failed", e)
