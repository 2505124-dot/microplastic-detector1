import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime


st.set_page_config(
    page_title="Microplastic Detector",
    page_icon="🔬",
    layout="wide"
)


st.title("🔬 Microplastic Screening Detector")

st.write(
    "ही प्रणाली microscope image मधील संभाव्य particles शोधते. "
    "हा laboratory-confirmed microplastic test नाही."
)


st.sidebar.header("Sample Information")

sample_type = st.sidebar.selectbox(
    "Water sample type निवडा",
    [
        "Tap water",
        "Bottled water",
        "Pond water",
        "Sea water",
        "Other"
    ]
)

sample_location = st.sidebar.text_input(
    "Sample location लिहा",
    placeholder="उदा. Ratnagiri"
)


st.sidebar.header("Detection Settings")

threshold_value = st.sidebar.slider(
    "Threshold value",
    min_value=50,
    max_value=240,
    value=180
)

min_area = st.sidebar.slider(
    "Minimum particle area",
    min_value=5,
    max_value=500,
    value=50
)

max_area = st.sidebar.slider(
    "Maximum particle area",
    min_value=500,
    max_value=10000,
    value=1000
)


uploaded_file = st.file_uploader(
    "Microscope image upload करा",
    type=["jpg", "jpeg", "png"]
)


analyze_button = st.button(
    "🔍 Analyze Image"
)


if uploaded_file is None:

    st.info(
        "Analysis सुरू करण्यासाठी JPG किंवा PNG image upload करा."
    )

    st.markdown(
        """
        ### वापरण्याची पद्धत

        1. Water sample type निवडा.
        2. Sample location लिहा.
        3. JPG किंवा PNG microscope image upload करा.
        4. `Analyze Image` button दाबा.
        5. Result आणि report download करा.
        """
    )

    st.stop()


if not analyze_button:

    st.info(
        "Image upload झाली आहे. Analysis सुरू करण्यासाठी "
        "`Analyze Image` button दाबा."
    )

    st.stop()


analysis_time = datetime.now().strftime(
    "%d-%m-%Y %H:%M"
)


file_bytes = np.asarray(
    bytearray(uploaded_file.getvalue()),
    dtype=np.uint8
)


image = cv2.imdecode(
    file_bytes,
    cv2.IMREAD_COLOR
)


if image is None:

    st.error(
        "Image read झाली नाही. JPG किंवा PNG image वापरून पुन्हा प्रयत्न करा."
    )

    st.stop()


gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


_, threshold = cv2.threshold(
    blur,
    threshold_value,
    255,
    cv2.THRESH_BINARY_INV
)


kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)


threshold = cv2.morphologyEx(
    threshold,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)


threshold = cv2.morphologyEx(
    threshold,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1
)


contours, _ = cv2.findContours(
    threshold,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


result = image.copy()
particle_count = 0
particle_details = []


for contour in contours:

    area = cv2.contourArea(contour)

    if min_area < area < max_area:

        x, y, width, height = cv2.boundingRect(
            contour
        )

        particle_count += 1

        particle_details.append(
            {
                "Particle": particle_count,
                "Area_pixels": round(area, 2),
                "X": x,
                "Y": y,
                "Width": width,
                "Height": height
            }
        )

        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (0, 0, 255),
            2
        )

        cv2.putText(
            result,
            "Possible particle",
            (x, max(y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )


st.success("Image analysis complete")


st.subheader("Sample Information")


info_col1, info_col2, info_col3 = st.columns(3)


with info_col1:

    st.write(
        f"**Sample type:** {sample_type}"
    )


with info_col2:

    st.write(
        f"**Location:** {sample_location or 'Not provided'}"
    )


with info_col3:

    st.write(
        f"**Analysis time:** {analysis_time}"
    )


st.subheader("Analysis Summary")


metric_col1, metric_col2, metric_col3 = st.columns(3)


with metric_col1:

    st.metric(
        "Possible particles",
        particle_count
    )


with metric_col2:

    st.metric(
        "Image width",
        f"{image.shape[1]} px"
    )


with metric_col3:

    st.metric(
        "Image height",
        f"{image.shape[0]} px"
    )


particle_df = pd.DataFrame(
    particle_details
)


st.subheader("Image Results")


tab1, tab2, tab3 = st.tabs(
    [
        "Detected Image",
        "Threshold Image",
        "Particle Report"
    ]
)


with tab1:

    result_rgb = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        result_rgb,
        caption="Possible particles marked with red boxes",
        use_container_width=True
    )


with tab2:

    st.image(
        threshold,
        caption="Processed threshold image",
        use_container_width=True
    )


with tab3:

    if not particle_df.empty:

        st.dataframe(
            particle_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Particle Area Chart")

        chart_data = particle_df.set_index(
            "Particle"
        )[["Area_pixels"]]

        st.bar_chart(
            chart_data
        )

    else:

        st.info(
            "या settings मध्ये कोणतेही possible particles सापडले नाहीत."
        )


if not particle_df.empty:

    report_df = particle_df.copy()

    report_df.insert(
        0,
        "Sample_type",
        sample_type
    )

    report_df.insert(
        1,
        "Location",
        sample_location or "Not provided"
    )

    report_df.insert(
        2,
        "Analysis_time",
        analysis_time
    )

    csv_data = report_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download particle report CSV",
        data=csv_data,
        file_name="particle_report.csv",
        mime="text/csv"
    )


result_success, result_encoded = cv2.imencode(
    ".jpg",
    result
)


threshold_success, threshold_encoded = cv2.imencode(
    ".jpg",
    threshold
)


st.subheader("Download Results")


download_col1, download_col2 = st.columns(2)


with download_col1:

    if result_success:

        st.download_button(
            label="Download result image",
            data=result_encoded.tobytes(),
            file_name="microplastic_result.jpg",
            mime="image/jpeg"
        )


with download_col2:

    if threshold_success:

        st.download_button(
            label="Download threshold image",
            data=threshold_encoded.tobytes(),
            file_name="threshold_image.jpg",
            mime="image/jpeg"
        )


st.warning(
    "महत्त्वाचे: हा result फक्त image-based screening आहे. "
    "प्रत्येक detected object microplastic असेलच असे नाही. "
    "अंतिम पुष्टीसाठी laboratory analysis आवश्यक आहे."
)