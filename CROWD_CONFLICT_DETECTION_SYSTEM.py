import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class MC_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.column1 = nn.Sequential(
            nn.Conv2d(3, 8, 9, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 7, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 16, 7, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 8, 7, padding='same'),
            nn.ReLU(),
        )

        self.column2 = nn.Sequential(
            nn.Conv2d(3, 10, 7, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(40, 20, 5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(20, 10, 5, padding='same'),
            nn.ReLU(),
        )

        self.column3 = nn.Sequential(
            nn.Conv2d(3, 12, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(48, 24, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(24, 12, 3, padding='same'),
            nn.ReLU(),
        )

        self.fusion_layer = nn.Sequential(
            nn.Conv2d(30, 1, 1, padding=0),
        )

    def forward(self, img_tensor):
        x1 = self.column1(img_tensor)
        x2 = self.column2(img_tensor)
        x3 = self.column3(img_tensor)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fusion_layer(x)
        return x

@st.cache_resource
def load_model():
    model = MC_CNN()
    model.load_state_dict(torch.load(r"C:\Users\Sanved\OneDrive\Desktop\Projects\ML\crowd_counting.pth", map_location=torch.device('cpu')), strict=False)
    model.eval()
    return model

# Load the model
model = load_model()

# Use the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to preprocess image
def preprocess_image(image_path, gt_downsample=4):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    ds_rows = int(img.shape[0] // gt_downsample)
    ds_cols = int(img.shape[1] // gt_downsample)
    img = cv2.resize(img, (ds_cols * gt_downsample, ds_rows * gt_downsample))
    img = img.transpose((2, 0, 1))
    img_tensor = torch.tensor(img / 255, dtype=torch.float)
    return img_tensor.unsqueeze(0)

# Function to allocate resources
def allocate_resources(pred_count, density_map):
    security_guard_thresholds = {50: 1, 100: 2, 200: 3, float('inf'): 4}
    medical_staff_thresholds = {150: 1, 300: 2, float('inf'): 3}
    num_security_guards = next(val for key, val in security_guard_thresholds.items() if pred_count <= key)
    num_medical_staff = next(val for key, val in medical_staff_thresholds.items() if pred_count <= key)

    density_map_np = density_map.cpu().squeeze().detach().numpy()
    rows, cols = density_map_np.shape
    north_density = np.sum(density_map_np[:rows // 2, :])
    south_density = np.sum(density_map_np[rows // 2:, :])
    west_density = np.sum(density_map_np[:, :cols // 2])
    east_density = np.sum(density_map_np[:, cols // 2:])

    # Logging density values
    print("North Density:", north_density)
    print("South Density:", south_density)
    print("West Density:", west_density)
    print("East Density:", east_density)

    # Visualize density map with annotations
    plt.imshow(density_map_np, cmap='jet')
    plt.title("Predicted Density Map")
    plt.colorbar(label='Density Value')
    plt.show()

    directions = {"North": north_density, "South": south_density, "West": west_density, "East": east_density}

    # Handle ties in maximum density direction
    max_density_value = max(directions.values())
    max_density_direction = [key for key, value in directions.items() if value == max_density_value]

    return {
        "Security Guards": num_security_guards,
        "Medical Staff": num_medical_staff,
        "High Density Direction": max_density_direction  # Changed to return list of directions
    }

# Streamlit UI starts here
st.title("Crowd Counting & Resource Allocation")
st.write("Upload an image and get crowd predictions along with security and medical staff recommendations.")

# Image uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        # Open the uploaded image directly
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert to NumPy array for processing
        img = np.array(image)
        img_path = 'temp_image.jpg'
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Preprocess and predict
        img_tensor = preprocess_image(img_path).to(device)
        with torch.inference_mode():
            pred_dm = model(img_tensor)

        # Calculate crowd count
        pred_count = int(np.maximum(0, pred_dm.sum().item()))
        st.write(f"Predicted number of people: *{pred_count}*")

        # Get predicted density map
        density_map_np = pred_dm.cpu().squeeze().detach().numpy()

        # Create a marker overlay
        image_ex = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # You might want to set a threshold to identify points of interest (heads)
        gt_coor_ex = np.argwhere(density_map_np > 0.1)  # Example threshold; adjust as necessary

        # Draw markers
        for x_cor, y_cor in gt_coor_ex:
            cv2.drawMarker(image_ex, (int(y_cor), int(x_cor)), (255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=3)

        # Display the image with markers
        plt.figure(figsize=(5, 5))
        plt.imshow(image_ex)
        plt.title("Image and Coordinate")
        st.pyplot(plt)

        # Display density map
        plt.figure(figsize=(5, 5))
        plt.imshow(density_map_np, cmap='jet')
        plt.title("Predicted Density Map")
        st.pyplot(plt)

        # Allocate resources
        resources = allocate_resources(pred_count, pred_dm)
        st.write(f"Security Guards Needed: *{resources['Security Guards']}*")
        st.write(f"Medical Staff Needed: *{resources['Medical Staff']}*")
        st.write(f"Area with Highest Density: *{', '.join(resources['High Density Direction'])}*")  # Display all directions

    except Exception as e:
        st.error(f"Error processing the image: {e}")
