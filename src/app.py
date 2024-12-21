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
    "Overview",
    "Transformations",
    "Scaling",
    "Denoising"
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

    if selected_page == "Overview":
        st.title("Overview")
        st.write("Here is the general information about the image.")
        height, width, depth = image.shape
        st.write(f"Height: {height} pixels")
        st.write(f"Width: {width} pixels")
        st.write(f"Depth: {depth} color components")

        # Create histograms for the Red, Green, and Blue channels
        colors = ['blue', 'green', 'red']
        channels = cv2.split(image) 

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

    elif selected_page == "Transformations":
        st.title("Transformations")
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


    elif selected_page == "Scaling":
        st.title("Scaling")
        st.write("Scaling consists of different image processing techniques that adjusts the size of an image. They include *rescaling, resizing, cropping* and *interpolation*.")
        # Scaling
        st.subheader("1. Rescaling")
        st.write("""
                *Definition*: Rescaling changes the size of an image by increasing or decreasing its dimensions proportionally. It is essentially resizing with a focus on maintaining the aspect ratio.
                
                *How It Works*: Rescaling works by multiplying the original dimensions (width and height) by a scaling factor:
                ```
                New Width = Original Width × Scale Factor
                New Height = Original Height × Scale Factor
                ```
                *Types*:
                - *Upscaling*: Enlarging the image (e.g., doubling its size).
                - *Downscaling*: Reducing the image size (e.g., halving its size).
                
                *Use Case*: Scaling is used to make images smaller for faster processing or larger for better visibility without worrying about exact dimensions.
                """)
        fx, fy = st.slider("Rescaling Factors (fx, fy)", 0.1, 2.0, (1.0, 1.0))
        rescaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)

        with col2:
            st.image(rescaled_image, caption=f"Scaled Image by {fx,fy}", channels="BGR", use_container_width=True)
            
        

        # Resizing
        st.subheader("2. Resizing")
        st.write("""
                *Definition*: Resizing explicitly changes the dimensions (width and height) of an image to specific values, regardless of its original aspect ratio.
                
                *How It Works*: The image is adjusted to fit the new dimensions, either preserving or distorting the original aspect ratio based on the resizing method.
                
                *Methods*:
                - *Keep Aspect Ratio*: Resize dimensions proportionally to maintain the original shape of the image.
                - *Ignore Aspect Ratio*: Resize to the specified dimensions, potentially distorting the image.
                
                *Use Case*: Resizing is crucial for preparing images for machine learning models, which often require fixed-size inputs.
                """)
        
        new_width = st.slider("New Width", 10, 1000, image.shape[1])
        new_height = st.slider("New Height", 10, 1000, image.shape[0])
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)

        with col2:
            st.image(resized_image, caption=f"Resized Image to {new_width, new_height}", channels="BGR", use_container_width=True)
        
        # Interpolation
        st.header("3. Interpolation")
        st.write("""
                *Definition*: Interpolation is the method used to estimate pixel values when resizing or scaling an image, especially for non-integer scaling factors.
                
                *How It Works*: Since resizing often involves creating new pixels, interpolation calculates their values based on nearby pixels. Different algorithms achieve this in various ways:
                - *Nearest-Neighbor Interpolation*: Assigns the value of the nearest pixel to the new pixel. Simple but can result in blocky images.
                - *Bilinear Interpolation*: Averages the values of the 2×2 surrounding pixels to estimate the new pixel value. Produces smoother images.
                - *Bicubic Interpolation*: Considers the 4×4 surrounding pixels for interpolation, producing even smoother results than bilinear.
                - *Lanczos Interpolation*: Uses a sinc function for high-quality resizing, particularly for downscaling.
                
                *Use Case*: Interpolation is necessary to prevent image distortion during resizing or scaling.

                """)
        
        # Choose interpolation method
        interpolation_method = st.selectbox(
            "**Choose Interpolation Method:**",
            ["Nearest Neighbor", "Bilinear", "Bicubic", "Lanczos"]
        )

        new_width = st.slider("New Width:", 50, 800, image.shape[1] // 2)
        new_height = st.slider("New Height:", 50, 800, image.shape[0] // 2)
    
        # Map selection to OpenCV constants
        interpolation_map = {
            "Nearest Neighbor": cv2.INTER_NEAREST,
            "Bilinear": cv2.INTER_LINEAR,
            "Bicubic": cv2.INTER_CUBIC,
            "Lanczos": cv2.INTER_LANCZOS4
        }
        chosen_interpolation = interpolation_map[interpolation_method]

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=chosen_interpolation)

        # Display the resized image
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)

        with col2:
            st.image(resized_image, caption=f"Resized Image ({interpolation_method})", channels="BGR", use_container_width=True)
            
        # Cropping
        st.subheader("4. Cropping")
        st.write("""
                *Definition*: Cropping removes unwanted outer areas of an image to focus on a specific region of interest (ROI).
                
                *How It Works*: You specify a rectangular region within the image, and only the pixels inside that rectangle are retained. The rest are discarded.
                
                *Use Case*: Cropping is used to zoom into specific parts of an image, remove unnecessary areas, or prepare smaller subregions for analysis.
                
                *Advantages*: It doesn’t alter the resolution or pixel quality of the retained portion of the image.
                
                *Drawbacks*: Cropping reduces the size of the image, potentially losing important contextual information outside the cropped area.

                """)
        x1, x2 = st.slider("Crop X-Axis Range", 0, image.shape[1], (0, image.shape[1]))
        y1, y2 = st.slider("Crop Y-Axis Range", 0, image.shape[0], (0, image.shape[0]))
        cropped_image = image[y1:y2, x1:x2]
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)

        with col2:
            st.image(cropped_image, caption="Cropped Image", channels="BGR", use_container_width=True)

    elif selected_page == "Denoising":
        st.title("Denoising")

        # Noise Removal
        st.write("""
                Denoising in image processing refers to the removal of noise from an image. Noise typically appears as random variations in brightness or color that degrade the image's quality. Noise can originate from various sources, such as low-light conditions, electronic interference, or sensor imperfections.
                
                Below are some of the common denoising techniques.
                """)

        # Averaging
        st.subheader("1. Averaging")
        st.write("Averaging replaces each pixel's value with the average of its neighbors. It reduces noise but may blur the image.")
        
        kernel_size_avg = st.slider("**Kernel Size for Averaging (Odd values only)**:", 3, 15, 5, step=2)
        averaged_image = cv2.blur(image, (kernel_size_avg, kernel_size_avg))
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)

        with col2:
            st.image(averaged_image, caption="Averaged Image", channels="BGR", use_container_width=True)

        # Median Filtering
        st.subheader("2. Median Filtering")
        st.write("Median Filtering replaces each pixel's value with the median value of the surrounding pixels. It is excellent for removing 'salt-and-pepper' noise while preserving edges.")
        
        kernel_size_med = st.slider("**Kernel Size for Median Filtering (Odd values only)**:", 3, 15, 5, step=2)
        median_filtered_image = cv2.medianBlur(image, kernel_size_med)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)

        with col2:
            st.image(median_filtered_image, caption="Median Filtered Image", channels="BGR", use_container_width=True)

        # Gaussian Filtering
        st.subheader("3. Guassian Filtering")
        st.write("It is a type of smoothing that uses a Gaussian kernel to reduce noise. Weighted averaging ensures smoother results without drastic edge distortion.")
        
        kernel_size_gauss = st.slider("**Kernel Size for Gaussian Filtering (Odd values only)**:", 3, 15, 5, step=2)
        gaussian_filtered_image = cv2.GaussianBlur(image, (kernel_size_gauss, kernel_size_gauss), 0)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)

        with col2:
            st.image(gaussian_filtered_image, caption="Gaussian Filtered Image", channels="BGR", use_container_width=True)

        # Non-Local Means (NLM) Filtering
        st.subheader("4. Non-Local Means (NLM) Filtering")
        st.write("This is an advanced method that denoises based on pixel similarity across the entire image, not just local neighborhoods. Balances noise reduction with edge and detail preservation.")
        
        h_nlm = st.slider("**Denoising Strength (h) for NLM**:", 5, 50, 10)
        nlm_filtered_image = cv2.fastNlMeansDenoisingColored(image, None, h_nlm, h_nlm, 7, 21)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", channels="BGR", use_container_width=True)

        with col2:
            st.image(nlm_filtered_image, caption="Non-Local Means (NLM) Filtered Image", channels="BGR", use_container_width=True)
