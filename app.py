import streamlit as st
# import langchain
import PyPDF2
import os
from transformers import BartTokenizer , BartForConditionalGeneration

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")


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

def generate_summary(text: str):
    # Tokenize the text
    tokens = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(tokens.input_ids, num_beams = 4, max_length = 200, early_stopping = True)


    return summary_ids




# Function to summarize text
def summarize_text(text: str) -> str:
    summary_ids = generate_summary(text)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_ip_tokenization_spaces=False)
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
    st.write("As of now supports only summarization.")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")

    if uploaded_file is not None:
        st.write("Paper uploaded successfully!")

        # Extract text from PDF
        text = extract_text_from_pdf(uploaded_file)

        # Display summary of the paper
        st.subheader("Summary of the Paper")
        with st.spinner("Brewing a potion for your paper's essence..."):
            summary = summarize_text(text)
            st.write(summary)

        # # Extract key information from the paper
        # st.subheader("Key Information")
        # paper_info = extract_paper_info(text)
        # st.write(paper_info)

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
