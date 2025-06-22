import matplotlib.pyplot as plt
import streamlit as st
import torch

from models import VAE


# Load model
@st.cache_resource
def load_model():
    vae = VAE(
        encoder_layer_sizes=[784, 256],
        latent_size=2,
        decoder_layer_sizes=[256, 784],
        conditional=True,
        num_labels=10,
    )
    state_dict = torch.load("vae.pt", map_location="cpu")
    vae.load_state_dict(state_dict)
    vae.eval()
    return vae


# Generate images from digit
def generate_images(model, digit, num_samples=8):
    device = torch.device("cpu")
    c = torch.full((num_samples,), digit, dtype=torch.long, device=device)

    z = torch.randn([c.size(0), 2]).to(device)
    with torch.no_grad():
        images = model.inference(z, c=c)
    return images.reshape(-1, 1, 28, 28)  # shape: (num_samples, 1, 28, 28)


# Streamlit UI
st.title("MNIST Digit Generator")

selected_digit = st.selectbox("Choose a digit (0-9)", list(range(10)))

model = load_model()

if st.button("Generate"):
    images = generate_images(model, selected_digit)

    # Plot 8 images
    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for ax, img in zip(axes, images):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.axis("off")
    st.pyplot(fig)
