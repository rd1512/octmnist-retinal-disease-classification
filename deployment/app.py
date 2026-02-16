import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import Improved_OCTMNIST_CNN

st.set_page_config(page_title="OCT Classification", layout="centered")
st.title("OCT Image Classification App")
st.write("Upload an OCT retinal image to classify the disease category.")

device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = Improved_OCTMNIST_CNN(num_classes=4)
    model.load_state_dict(torch.load("improved_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

class_names = [
    'CHOROIDAL NEOVASCULARIZATION',
    'DIABETIC MACULAR EDEMA',
    'DRUSEN',
    'NORMAL'
]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("Choose an OCT image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Uploaded Image", width="stretch")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    st.subheader("Prediction:")
    st.success(f"Predicted Class: {class_names[predicted_class]}")

    st.subheader("Confidence Scores:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {probabilities[0][i].item()*100:.2f}%")