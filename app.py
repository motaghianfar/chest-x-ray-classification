import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
from model.chexnet_model import create_chexnet_model
from utils.preprocessing import preprocess_image, get_chest_xray_labels

# Set page config
st.set_page_config(
    page_title="CheXNet Chest X-ray Analysis",
    page_icon="🩻",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the CheXNet model"""
    try:
        weights_path = "weights/chexnet_weights.pth"
        model = create_chexnet_model(weights_path=weights_path)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("🩻 CheXNet Chest X-ray Analysis")
    st.write("Upload a chest X-ray image to detect 14 different pathologies")
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check if weights file exists.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
        
        with col2:
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    try:
                        # Preprocess image
                        input_tensor = preprocess_image(image)
                        
                        # Make prediction
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probabilities = outputs.cpu().numpy()[0]
                        
                        # Get labels
                        labels = get_chest_xray_labels()
                        
                        # Display results
                        st.subheader("Detection Results")
                        
                        # Create results table
                        results = []
                        for i, (label, prob) in enumerate(zip(labels, probabilities)):
                            results.append({
                                "Pathology": label,
                                "Probability": f"{prob:.3f}",
                                "Confidence": prob
                            })
                        
                        # Sort by confidence
                        results.sort(key=lambda x: x["Confidence"], reverse=True)
                        
                        # Display top predictions
                        for i, result in enumerate(results[:5]):
                            prob = float(result["Probability"])
                            color = "red" if prob > 0.5 else "orange" if prob > 0.3 else "green"
                            st.markdown(
                                f"**{i+1}. {result['Pathology']}**: "
                                f"<span style='color:{color}'>{prob:.1%}</span>", 
                                unsafe_allow_html=True
                            )
                        
                        # Show all results in expander
                        with st.expander("View All Results"):
                            for result in results:
                                prob = float(result["Probability"])
                                color = "red" if prob > 0.5 else "orange" if prob > 0.3 else "green"
                                st.markdown(
                                    f"- {result['Pathology']}: "
                                    f"<span style='color:{color}'>{prob:.1%}</span>", 
                                    unsafe_allow_html=True
                                )
                                
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")

    # Add information section
    with st.expander("About CheXNet"):
        st.markdown("""
        **CheXNet** is a 121-layer convolutional neural network that:
        - Detects 14 common chest pathologies from X-ray images
        - Uses DenseNet-121 architecture
        - Provides probability scores for each condition
        
        **Note**: This tool is for educational/demonstration purposes only. 
        Always consult healthcare professionals for medical diagnosis.
        """)

if __name__ == "__main__":
    main()
