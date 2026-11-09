from src.inference.model_loader import load_keras_model

path = "models/mwcd_materials/mobilenet_20261029T021239Z.keras"

model = load_keras_model(path)

print("Model load thành công!")
print("Input:", model.input_shape)
print("Output:", model.output_shape)

model.summary()