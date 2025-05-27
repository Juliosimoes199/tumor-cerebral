import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Detecção de Deficiência de Nutrientes em Folhas de Alface")

# Carregue o modelo treinado (isso será feito apenas uma vez)
@st.cache_resource
def load_model():
    return YOLO('/Users/PC/Documents/Estágio_Tecn_Socity/deficiencia de nutrientes em folhas de alface/Lettuce-NPK-Yolov11/runs/detect/train/weights/best.pt')

model = load_model()

# Widget para fazer o upload da imagem
uploaded_file = st.file_uploader("Faça o upload de uma imagem de folha de alface...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Exibe a imagem carregada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem Carregada", use_container_width=True)

    # Botão para realizar a detecção
    if st.button("Detectar Deficiências"):
        with st.spinner("Analisando a imagem..."):
            # Realiza as previsões na imagem carregada
            results = model.predict(image)

            for result in results:
                boxes = result.boxes  # Caixas delimitadoras (x1, y1, x2, y2)
                classes = boxes.cls
                confidences = boxes.conf  # Pontuações de confiança
                xyxy = boxes.xyxy

                st.subheader("Resultados da Detecção:")
                if len(classes) > 0:
                    st.write("Classes detectadas:", classes.cpu().numpy())
                    st.write("Pontuações de confiança:", confidences.cpu().numpy())
                    st.write("Coordenadas das caixas delimitadoras:", xyxy.cpu().numpy())

                    # Exibe a imagem com as caixas delimitadoras
                    annotated_image = result.plot()
                    st.image(annotated_image, caption="Imagem com Detecções", use_container_width=True)
                else:
                    st.write("Nenhuma deficiência de nutriente detectada nesta imagem.")