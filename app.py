
import streamlit as st
import cv2
import numpy as np
import pytesseract
import fitz

from PIL import Image
from io import BytesIO

from scanner import (
    detect_document,
    crop_document,
    enhance_document
)

from ai_features import (
    generate_summary,
    generate_easy_notes,
    generate_quiz,
    generate_flashcards,
    generate_ppt_content
)

from ppt_generator import create_presentation

# PAGE CONFIGURATION
st.set_page_config(
    page_title="DOC AI | Smart Study Assistant",
    page_icon="📄",
    layout="wide"
)

# TESSERACT LOCATION
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# PREMIUM DARK CSS
st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: #F1F5F9;
}

[data-testid="stSidebar"] {
    background-color: #111B2E;
    border-right: 1px solid #26354D;
}

h1, h2, h3 {
    color: #F1F5F9 !important;
}

p, label {
    color: #CBD5E1;
}

.stButton > button {
    background: linear-gradient(90deg, #2563EB, #6366F1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1D4ED8, #4F46E5);
    color: white;
    border: none;
}

.stDownloadButton > button {
    background-color: #0F766E;
    color: white;
    border: none;
    border-radius: 10px;
    width: 100%;
}

div[data-testid="stFileUploader"] {
    background-color: #172338;
    border-radius: 12px;
    padding: 10px;
}

div[data-testid="stExpander"] {
    background-color: #152238;
    border: 1px solid #26354D;
    border-radius: 10px;
}

[data-testid="stMetric"] {
    background-color: #172338;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #26354D;
}

hr {
    border-color: #26354D;
}

</style>
""", unsafe_allow_html=True)

# PDF TEXT EXTRACTION
def extract_pdf_text(pdf_file):

    document = fitz.open(
        stream=pdf_file.getvalue(),
        filetype="pdf"
    )

    all_text = ""

    for page_number, page in enumerate(document):

        page_text = page.get_text().strip()

        if not page_text:

            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False
            )

            page_image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            page_text = pytesseract.image_to_string(
                page_image,
                lang="eng"
            )

        all_text += (
            f"\n--- Page {page_number + 1} ---\n"
            + page_text
        )

    document.close()

    return all_text

# SESSION STATE
if "pdf_text" not in st.session_state:
    st.session_state["pdf_text"] = ""

# SIDEBAR
with st.sidebar:

    st.markdown("# 📄 DOC AI")
    st.caption("Smart Document & Study Assistant")

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Document Scanner",
            "PDF Extractor",
            "Image OCR",
            "AI Summary",
            "Easy Study Notes",
            "Quiz Generator",
            "Flashcards",
            "PPT Generator"
        ]
    )

    st.divider()

    st.markdown("### 📂 Study Document")

    uploaded_pdf = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        key="main_pdf"
    )

    if st.button("Extract PDF Text"):

        if uploaded_pdf:

            with st.spinner("Reading your document..."):

                try:
                    extracted = extract_pdf_text(uploaded_pdf)

                    st.session_state["pdf_text"] = extracted

                    st.success("Text extracted successfully!")

                except Exception as e:
                    st.error(f"Extraction failed: {e}")

        else:
            st.warning("Please upload a PDF first.")

    if st.session_state["pdf_text"]:
        st.success("Document is ready for AI features.")
    else:
        st.info("No document extracted yet.")

    st.divider()

    st.caption("Built with Python, Streamlit and AI")

# MAIN HEADER
st.markdown(
    "<h1 style='color:#8BB8FF;'>DOC AI</h1>",
    unsafe_allow_html=True
)

st.caption("YOUR SMART DOCUMENT & STUDY WORKSPACE")

st.divider()

# DASHBOARD
if page == "Dashboard":

    st.title("Welcome to Your AI Workspace 👋")

    st.write(
        "Convert your documents into smart study material."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AI Features", "5")

    with col2:
        st.metric("Document Tools", "3")

    with col3:
        st.metric(
            "Document Status",
            "Ready" if st.session_state["pdf_text"] else "Waiting"
        )

    st.subheader("Explore Your Features")

    features = [
        ("📷", "Document Scanner", "Scan and enhance document images."),
        ("📚", "PDF Extractor", "Extract text from normal and scanned PDFs."),
        ("🤖", "AI Summary", "Understand long documents quickly."),
        ("📝", "Easy Study Notes", "Convert content into simple notes."),
        ("❓", "Quiz Generator", "Generate MCQs for exam preparation."),
        ("🗂️", "Flashcards", "Create question-answer study cards."),
        ("📊", "PPT Generator", "Create PowerPoint presentations.")
    ]

    for icon, name, description in features:

        with st.container(border=True):
            st.markdown(f"### {icon} {name}")
            st.write(description)

    st.info(
        "Tip: Upload a PDF from the sidebar and extract its text "
        "before using AI features."
    )

# DOCUMENT SCANNER
elif page == "Document Scanner":

    st.title("📷 Document Scanner")

    st.write(
        "Upload document images to crop and enhance them."
    )

    uploaded_images = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_images:

        scanned_images = []

        for uploaded_file in uploaded_images:

            image_bytes = np.asarray(
                bytearray(uploaded_file.read()),
                dtype=np.uint8
            )

            image = cv2.imdecode(
                image_bytes,
                cv2.IMREAD_COLOR
            )

            corners, edges = detect_document(image)

            if corners is not None:
                scanned = crop_document(image, corners)
            else:
                scanned = image

            enhanced = enhance_document(scanned)

            scanned_images.append(enhanced)

            st.image(
                cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB),
                caption=uploaded_file.name,
                use_container_width=True
            )

        if st.button("Create Combined PDF"):

            pdf_images = []

            for scanned in scanned_images:

                rgb_image = cv2.cvtColor(
                    scanned,
                    cv2.COLOR_BGR2RGB
                )

                pdf_images.append(
                    Image.fromarray(rgb_image)
                )

            if pdf_images:

                pdf_buffer = BytesIO()

                pdf_images[0].save(
                    pdf_buffer,
                    format="PDF",
                    save_all=True,
                    append_images=pdf_images[1:]
                )

                st.download_button(
                    "Download Scanned PDF",
                    data=pdf_buffer.getvalue(),
                    file_name="scanned_document.pdf",
                    mime="application/pdf"
                )

# PDF EXTRACTOR
elif page == "PDF Extractor":

    st.title("📚 PDF Extractor & OCR")

    st.write(
        "Extract selectable text or use OCR on scanned PDF pages."
    )

    if st.session_state["pdf_text"]:

        st.success("PDF text is available.")

        with st.expander("View Extracted Text"):
            st.text_area(
                "Extracted Text",
                st.session_state["pdf_text"],
                height=350
            )

        st.download_button(
            "Download Extracted Text",
            data=st.session_state["pdf_text"],
            file_name="extracted_text.txt",
            mime="text/plain"
        )

    else:
        st.info(
            "Upload your PDF and click Extract PDF Text "
            "in the sidebar."
        )

# IMAGE OCR
elif page == "Image OCR":

    st.title("🔍 Image Text Extractor")

    ocr_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="ocr_image"
    )

    if ocr_image:

        image = Image.open(ocr_image)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("Extract Image Text"):

            with st.spinner("Reading image..."):

                try:

                    extracted = pytesseract.image_to_string(
                        image,
                        lang="eng"
                    )

                    st.subheader("Extracted Text")
                    st.write(extracted)

                    st.download_button(
                        "Download Text",
                        data=extracted,
                        file_name="image_text.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"OCR failed: {e}")

# AI SUMMARY
elif page == "AI Summary":

    st.title("🤖 AI Summary Generator")

    if st.button("Generate Summary"):

        if st.session_state["pdf_text"]:

            with st.spinner("Generating summary..."):

                try:

                    result = generate_summary(
                        st.session_state["pdf_text"]
                    )

                    st.subheader("Your Summary")
                    st.write(result)

                except Exception as e:
                    st.error(f"Summary failed: {e}")

        else:
            st.warning("Extract PDF text first.")

# EASY NOTES
elif page == "Easy Study Notes":

    st.title("📝 Easy Study Notes")

    if st.button("Generate Easy Notes"):

        if st.session_state["pdf_text"]:

            with st.spinner("Creating study notes..."):

                try:

                    result = generate_easy_notes(
                        st.session_state["pdf_text"]
                    )

                    st.subheader("Your Study Notes")
                    st.write(result)

                except Exception as e:
                    st.error(f"Notes failed: {e}")

        else:
            st.warning("Extract PDF text first.")

# QUIZ GENERATOR
elif page == "Quiz Generator":

    st.title("❓ AI Quiz Generator")

    st.write("Create 10 MCQs from your uploaded document.")

    if st.button("Generate Quiz"):

        if st.session_state["pdf_text"]:

            with st.spinner("Generating quiz..."):

                try:

                    result = generate_quiz(
                        st.session_state["pdf_text"]
                    )

                    st.subheader("Your Quiz")
                    st.write(result)

                except Exception as e:
                    st.error(f"Quiz failed: {e}")

        else:
            st.warning("Extract PDF text first.")

# FLASHCARDS
elif page == "Flashcards":

    st.title("🗂️ AI Flashcards Generator")

    st.write(
        "Create question-answer flashcards from your document."
    )

    if st.button("Generate Flashcards"):

        if st.session_state["pdf_text"]:

            with st.spinner("Creating flashcards..."):

                try:

                    result = generate_flashcards(
                        st.session_state["pdf_text"]
                    )

                    st.subheader("Your Flashcards")
                    st.write(result)

                except Exception as e:
                    st.error(f"Flashcards failed: {e}")

        else:
            st.warning("Extract PDF text first.")

# PPT GENERATOR
elif page == "PPT Generator":

    st.title("📊 AI PowerPoint Generator")

    st.write(
        "Convert your study document into a PowerPoint presentation."
    )

    if st.button("Generate PowerPoint"):

        if st.session_state["pdf_text"]:

            with st.spinner("Creating your presentation..."):

                try:

                    ppt_data = generate_ppt_content(
                        st.session_state["pdf_text"]
                    )

                    ppt_file = create_presentation(ppt_data)

                    st.success("Presentation created successfully!")

                    st.download_button(
                        "Download PowerPoint",
                        data=ppt_file.getvalue(),
                        file_name="AI_Study_Presentation.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )

                except Exception as e:
                    st.error(f"PPT generation failed: {e}")

        else:
            st.warning("Extract PDF text first.")

# FOOTER
st.divider()

st.markdown(
    "<p style='text-align:center;color:#8193AE;'>"
    "DOC AI | AI Document Scanner & Study Assistant"
    "<br>Built with Python and Streamlit"
    "</p>",
    unsafe_allow_html=True
)