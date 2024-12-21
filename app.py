import streamlit as st 
from PIL import Image
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Header
st.title("Image-PRO")

st.write("""This is the ultimate **image processing master**. 
            It displays for you how that image will look like when you apply the different processing and manipulation techniques.
            Do you have an image that you want to try out with this **imagepro**?  
            *Upload it and see the magic!*""")
st.markdown("---")

# Side bar
st.sidebar.title("Image Pro")
st.sidebar.subheader("Navigation")

pages = [
    "Basic Analysis",
    "Translation, Rotation, Transpose & Flip",
    "Scaling, Resizing, Interpolation & Cropping",
    "Drawing on Images",
    "Noise Removal & Filters"
]
selected_page = st.sidebar.radio("**Select A Technique:**", pages)

upload_file = st.sidebar.file_uploader("Upload your image", type=["jpg","png","jpeg"])
alert_placeholder = st.empty()

# Image state management
if "image_data" not in st.session_state:
    st.session_state.image_data = None

if upload_file is not None:
    pil_image = Image.open(upload_file)
    st.session_state.image_data = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    st.image(pil_image, caption="original image", use_container_width=True)
    with alert_placeholder.container():
        st.success("Image uploaded successfully!")
    time.sleep(3)
    alert_placeholder.empty()

# Show error if no image is uploaded
if st.session_state.image_data is None:
    st.error("Please upload an image to proceed.")
else:
    # Display the image
    st.sidebar.image(st.session_state.image_data, caption="Uploaded Image", channels="BGR", use_container_width=True)

    # Navigation logic
    image = st.session_state.image_data

    if selected_page == "Basic Analysis":
        st.title("Basic Image Analysis")
        height, width, depth = image.shape
        st.write(f"Height: {height} pixels")
        st.write(f"Width: {width} pixels")
        st.write(f"Depth: {depth} color components")

        # Create histograms for the Red, Green, and Blue channels
        colors = ['blue', 'green', 'red']
        channels = cv2.split(image)  # Split the image into its BGR channels

        # Create an empty list for the plotly traces
        traces = []

        # Plot histograms for each channel
        for i, color in enumerate(colors):
            hist = cv2.calcHist([channels[i]], [0], None, [256], [0, 256])
            trace = go.Scatter(
                x=np.arange(256),
                y=hist.flatten(),
                mode='lines',
                name=f'{color.capitalize()} Channel',
                line=dict(color=color)
            )
            traces.append(trace)

        # Create the layout for the plot
        layout = go.Layout(
            title="Histogram of RGB Channels",
            xaxis=dict(title="Pixel Intensity"),
            yaxis=dict(title="Frequency"),
            showlegend=True
        )

        # Create the figure and display it
        fig = go.Figure(data=traces, layout=layout)
        st.plotly_chart(fig)
    
        st.subheader("Grayscale Conversion")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)

        with col2:
            st.image(gray_image, caption="Grayscale Image", channels="GRAY", use_container_width=True)
            
        st.subheader("Histogram")
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        st.bar_chart(hist.flatten())

    elif selected_page == "Translation, Rotation, Transpose & Flip":
        st.title("Translation, Rotation, Transpose & Flip")
        st.write("""
                    In image processing, **translation, rotation, transpose**, and **flip** are common geometric transformations that manipulate the position and orientation of an image. 
                    These transformations are essential for various tasks, such as image augmentation, alignment, and data manipulation in computer vision.""")
                    
        # Translation
        st.subheader("1. Translation")
        st.write("""
                *Definition*: Translation involves shifting the position of an image along the X and Y axes (horizontal and vertical) without changing its orientation or shape.
                
                *How It Works*: Each pixel in the image is moved by a specified amount in the X and Y directions. This is typically done by adding an offset to the coordinates of each pixel.
                
                *Mathematical Representation*: The translation of an image can be represented as:
                
                `New Coordinates=(Original X+tx,Original Y+ty)`, where *tx* and *ty* are the translation distances along the X and Y axes, respectively.
                
                *Use Case*: Translation is commonly used in tasks like shifting an image to create data augmentation or when aligning multiple images in an image stitching task. 
                """)
        st.markdown("---")
        tx, ty = st.slider("Translation Offset (x, y)", -100, 100, (0, 0))
        M_translation = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, M_translation, (image.shape[1], image.shape[0]))
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)
        with col2:
            st.image(translated_image, caption=f"Translated Image with offset: {tx,ty}", channels="BGR", use_container_width=True)

        # Rotation
        st.subheader("2. Rotation")
        st.write("""
                *Definition*: Rotation involves turning an image by a certain angle, usually around its center. The image is rotated in a counterclockwise direction by default.
                
                *How It Works*: Each pixel is repositioned based on the rotation angle, with the center of the image typically serving as the pivot point. The new position of a pixel can be calculated using the following formulas:
                
                ```
                x′ = x⋅cos(θ) − y⋅sin(θ)
                𝑦′ = 𝑥⋅sin(𝜃) + 𝑦⋅cos(𝜃)
                ```
                where:
                - *(x,y)* are the coordinates of a pixel before rotation,
                - *(x′,y′)* are the new coordinates,
                - *θ* is the rotation angle.
                
                *Use Case*: Rotation is often used in image augmentation to generate rotated versions of an image or to align images for processing.
                """)
        st.markdown("---")
        angle = st.slider("Rotation Angle", -180, 180, 0)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M_rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M_rotation, (image.shape[1], image.shape[0]))
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)
        with col2:
            st.image(rotated_image, caption=f"Rotated Image at {angle} degrees", channels="BGR", use_container_width=True)

        # Transpose
        st.subheader("3. Transpose")
        st.write("""
                *Definition*: Transposing an image means swapping its rows and columns. In other words, it flips the image over its diagonal (from top-left to bottom-right).
                
                *How It Works*: The pixel at position *(i, j)* in the original image is moved to position *(j, i)* in the transposed image.
                
                *Use Case*: Transposing is useful in mathematical operations such as matrix manipulation and for certain image processing algorithms that require a change in image orientation.
                """)
        transposed_image = cv2.transpose(image)
        
        # Display original and flipped images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)
        with col2:
            st.image(transposed_image, caption="Transposed Image", channels="BGR", use_container_width=True)

        # Flip
        st.subheader("4. Flip")
        st.write("""
                *Definition*: Flipping an image refers to reflecting it along a specific axis (horizontal or vertical). There are two common types of flips:
                
                - *Horizontal Flip (Left-Right Flip)*: Reflects the image along the vertical axis (left-to-right flip).
                
                - *Vertical Flip (Top-Bottom Flip)*: Reflects the image along the horizontal axis (top-to-bottom flip).
                    
                *How It Works*:
                
                - *Horizontal Flip*: The pixel at position (x, y) moves to position (width - x, y).
                
                - *Vertical Flip*: The pixel at position (x, y) moves to position (x, height - y).
                    
                *Use Case*: Flipping is commonly used in data augmentation for training machine learning models, or in tasks like mirroring images in graphics applications.
                """)
        st.markdown("---")
        # Flip mode selection
        flip_option = st.radio("Flip Mode:", ["Horizontal", "Vertical", "Both"])
        flip_code = {"Horizontal": 1, "Vertical": 0, "Both": -1}[flip_option]
        flipped_image = cv2.flip(image, flip_code)

        # Display original and flipped images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)

        with col2:
            st.image(flipped_image, caption=f"Flipped Image ({flip_option})", channels="BGR", use_container_width=True)


    elif selected_page == "Scaling, Resizing, Interpolation & Cropping":
        st.title("Scaling, Resizing, Interpolation & Cropping")

        # Scaling
        st.subheader("1. Scaling")
        st.write("""
                *Definition*: Scaling changes the size of an image by increasing or decreasing its dimensions proportionally. It is essentially resizing with a focus on maintaining the aspect ratio.
                
                *How It Works*: Scaling works by multiplying the original dimensions (width and height) by a scaling factor:
                ```
                New Width = Original Width × Scale Factor
                New Height = Original Height × Scale Factor
                ```
                *Types*:
                - *Upscaling*: Enlarging the image (e.g., doubling its size).
                - *Downscaling*: Reducing the image size (e.g., halving its size).
                
                *Use Case*: Scaling is used to make images smaller for faster processing or larger for better visibility without worrying about exact dimensions.
                """)
        fx, fy = st.slider("Scaling Factors (fx, fy)", 0.1, 2.0, (1.0, 1.0))
        scaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)

        with col2:
            st.image(scaled_image, caption=f"Scaled Image by {fx,fy}", channels="BGR", use_container_width=True)
            
        

        # Resizing
        st.subheader("Resizing")
        new_width = st.slider("New Width", 10, 1000, image.shape[1])
        new_height = st.slider("New Height", 10, 1000, image.shape[0])
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)

        with col2:
            st.image(resized_image, caption=f"Resized Image to {new_width, new_height}", channels="BGR", use_container_width=True)
        
        # Cropping
        st.subheader("Cropping")
        x1, x2 = st.slider("Crop X-Axis Range", 0, image.shape[1], (0, image.shape[1]))
        y1, y2 = st.slider("Crop Y-Axis Range", 0, image.shape[0], (0, image.shape[0]))
        cropped_image = image[y1:y2, x1:x2]
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)

        with col2:
            st.image(cropped_image, caption="Cropped Image", channels="BGR", use_container_width=True)

    elif selected_page == "Drawing on Images":
        st.title("Drawing on Images")

        # Drawing shapes
        st.subheader("Draw Shapes")
        shape_option = st.radio("Shape", ["Rectangle", "Circle", "Line"])
        color = st.color_picker("Pick a Color", "#00f900")
        color_rgb = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

        if shape_option == "Rectangle":
            x1, y1 = st.slider("Top-Left Corner (x1, y1)", 0, 500, (50, 50))
            x2, y2 = st.slider("Bottom-Right Corner (x2, y2)", 0, 500, (200, 200))
            temp_image = image.copy()
            cv2.rectangle(temp_image, (x1, y1), (x2, y2), color_rgb, -1)
            st.image(temp_image, caption="Rectangle", channels="BGR", use_container_width=True)

        elif shape_option == "Circle":
            cx, cy = st.slider("Center (cx, cy)", 0, 500, (100, 100))
            radius = st.slider("Radius", 1, 200, 50)
            temp_image = image.copy()
            cv2.circle(temp_image, (cx, cy), radius, color_rgb, -1)
            st.image(temp_image, caption="Circle", channels="BGR", use_container_width=True)

        elif shape_option == "Line":
            x1, y1 = st.slider("Start Point (x1, y1)", 0, 500, (50, 50))
            x2, y2 = st.slider("End Point (x2, y2)", 0, 500, (200, 200))
            temp_image = image.copy()
            cv2.line(temp_image, (x1, y1), (x2, y2), color_rgb, 5)
            st.image(temp_image, caption="Line", channels="BGR", use_container_width=True)

    elif selected_page == "Noise Removal & Filters":
        st.title("Noise Removal & Filters")

        # Noise Removal
        st.subheader("Denoising")
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        st.image(denoised_image, caption="Denoised Image", channels="BGR", use_container_width=True)

        # Gaussian Blur
        st.subheader("Gaussian Blur")
        ksize = st.slider("Kernel Size", 1, 21, 5, step=2)
        blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        st.image(blurred_image, caption="Blurred Image", channels="BGR", use_container_width=True)