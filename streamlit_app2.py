import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

# Remove unnecessary imports and warnings suppression for clarity

st.set_page_config(
    page_title="Sushi or Steak or Pizza?!",
    page_icon=":girl:",
    initial_sidebar_state='auto'
)

# Style hiding is fine, keeping it as is
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.sidebar:
        #st.image('mg.png')
        st.title(":sushi:画像識別:pizza:")
        st.subheader("Resnetを転移学習したモデルを使用しています。Sushi:sushi: or Steak:meat_on_bone: or Pizza:pizza:?!を判別します:rainbow:")
        st.subheader("Github: https://github.com/hayata1996/stsushidetectionapp")
        #st.subheader("トレーニングに使用した画像リンク: https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset?ref=blog.streamlit.io")


@st.cache_resource
def load_model():
    model_save_path = 'model.pth'
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    NUM_CLASSES = 3
    model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
    
    checkpoint = torch.load(model_save_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.eval()

model = load_model()

st.title("ピザか寿司かステーキの画像をアップロードしてみてください:muscle:")

file = st.file_uploader("", type=["jpg", "png"])

def import_and_predict(image_data, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_data)
    image = transform(image)  # 画像を変換
    
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # モデルへの入力を準備
        probabilities = F.softmax(output, dim=1)[0]  # Softmaxとバッチ次元の削除

    return output[0], probabilities  # 最初のバッチの結果のみを返す

if file is not None:
    st.image(file, use_column_width=True)
    predictions, probabilities = import_and_predict(file, model)
    _, predicted_class = torch.max(predictions, 0)
    predicted_class = predicted_class.item()
    predicted_probability = probabilities.max().item()

    class_names = ['Pizza', 'Steak', 'Sushi']
    string = f"これは...: {class_names[predicted_class]}である確率{predicted_probability:.2f}!!"

    st.sidebar.error(string)

    # 各クラスの確率を表示
    st.sidebar.success(f"Pizza: {probabilities[0]:.2f}, Steak: {probabilities[1]:.2f}, Sushi: {probabilities[2]:.2f}")
