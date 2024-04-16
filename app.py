import streamlit as st
from transformers import pipeline
import langchain
import PyPDF2
import os


def save_uploaded_file(uploaded_file):
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    RP_file = save_uploaded_file(pdf_file)
    with open(RP_file, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


# Function to summarize text
def summarize_text(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary


# Function to extract key information from the paper
def extract_paper_info(text):
    # Logic to extract key information from the paper (e.g., using regex, NLP techniques)
    # This part can be expanded based on the specific requirements
    pass


# Function to build and fine-tune the chatbot
def build_chatbot():
    # Fine-tuning language model for chatbot using Langchain
    lang_model = ''

    # Additional fine-tuning steps can be added here

    return lang_model


# Main function to run the Streamlit app
def main():
    st.title("Research Paper Understanding Chatbot")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")

    if uploaded_file is not None:
        st.write("Paper uploaded successfully!")

        # Extract text from PDF
        text = extract_text_from_pdf(uploaded_file)

        # Display summary of the paper
        st.subheader("Summary of the Paper")
        summary = summarize_text(text)
        st.write(summary)

        # Extract key information from the paper
        st.subheader("Key Information")
        paper_info = extract_paper_info(text)
        st.write(paper_info)

        # # Build chatbot
        # st.subheader("Chatbot")
        # chatbot = build_chatbot()

        # # Chat interface
        # user_input = st.text_input("You: ")
        # if user_input:
        #     response = chatbot.generate_response(user_input)
        #     st.write("Chatbot:", response)

    else:
        st.write("Please upload a PDF file")


if __name__ == "__main__":
    main()
