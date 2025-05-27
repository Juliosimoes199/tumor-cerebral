import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# Configurações da página
st.set_page_config(
    page_title="Detecção de tumor cerebral",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Barra lateral para configurações e informações
with st.sidebar:
    st.title("⚙️ Configurações")
    st.markdown("Aplicação para detecção de tumor cerebral com Visão Computacional.")
    st.markdown("---")
    st.subheader("Instruções")
    st.markdown("1. Faça o upload de uma imagem de Ressonância Magnética (RM) do cérebro.")
    st.markdown("2. Clique no botão 'Examinar'.")
    st.markdown("3. Os resultados do exames serão exibidos abaixo.")
    st.markdown("---")
    st.subheader("Sobre o Modelo")
    st.markdown("Modelo YOLOv11 treinado para identificar a criação de tumor a partir de imagens de RM.")
    st.markdown("---")
    st.info("Desenvolvido pela Tecno Society.")

# Conteúdo principal
st.title("🧠 Detecção de Tumor Cerebral com imagens de Ressonância Magnética")
st.markdown("---")

# Carregue o modelo treinado (isso será feito apenas uma vez)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Widget para fazer o upload da imagem
uploaded_file = st.file_uploader("Faça o upload de uma imagem de Ressonância Magnética do Cérebro...", type=["jpg", "jpeg", "png"])

st.markdown("---")

if uploaded_file is not None:
    # Layout em colunas para a imagem carregada e o botão
    col1, col2 = st.columns([1, 1])

    with col1:
        try:
            # Exibe a imagem carregada
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem Carregada", use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao abrir a imagem: {e}")

    with col2:
        detect_button = st.button("Examinar")

    # Botão para realizar a detecção (centralizado abaixo da imagem)
    if detect_button:
        if uploaded_file is not None:
            with st.spinner("Analisando a imagem..."):
                try:
                    # Realiza as previsões na imagem carregada
                    image = Image.open(uploaded_file)
                    results = model.predict(image)

                    st.subheader("Resultados do Exame:")
                    if results and results[0].boxes:
                        result = results[0]
                        boxes = result.boxes
                        classes = boxes.cls
                        confidences = boxes.conf
                        xyxy = boxes.xyxy

                        st.write("Classes detectadas:", classes.cpu().numpy())
                        st.write("Pontuações de confiança:", confidences.cpu().numpy())
                        st.write("Coordenadas das caixas delimitadoras:", xyxy.cpu().numpy())

                        # Exibe a imagem com as caixas delimitadoras
                        annotated_image = results[0].plot()
                        st.image(annotated_image, caption="Imagem com Detecções", use_container_width=True)
                    else:
                        st.info("Nenhum Possível Tumor Detectado nesta imagem.")
                except Exception as e:
                    st.error(f"Erro durante a detecção: {e}")
            st.markdown("---")
