import streamlit as st
import os
from dotenv import load_dotenv
from utils import extract_text_from_pdf, chunk_text, extract_keywords
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.documents import Document

# Loading the environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

st.set_page_config(page_title="PDF Summarizer", layout="wide")
st.title("PDF Summarizer + Keyword Insights")

uploaded_file = st.file_uploader("Upload your PDF File", type=["pdf"])

# Summarization prompt
def summarize_text_with_prompt(llm, docs):
    prompt_template = PromptTemplate.from_template(
        """You are a helpful assistant. Read the following text extracted from a PDF document and generate a clear, concise, and informative summary.

Your summary should:
- Capture the main topic and purpose of the document
- Highlight key points, facts, or arguments
- Mention any examples, data, or conclusions if present
- Be written in plain, natural language that anyone can understand

Text:
{input}

Summary:"""
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    full_text = "\n\n".join(doc.page_content for doc in docs)
    return chain.invoke({"input": full_text})

# Keyword Prompt
def refine_and_explain_keywords(llm, keywords, context):
    prompt_template = PromptTemplate.from_template(
        """You are a helpful assistant. The following keywords were extracted from a document using a statistical method (YAKE), but some may be too generic or repetitive.

Please:
1. Remove any vague, generic, or duplicate terms.
2. Return a refined list of meaningful keywords.
3. Provide a short, beginner-friendly explanation for each keyword based on the document context.

Document:
{context}

Raw Keywords:
{keywords}

Refined Keywords with Explanations:"""
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.invoke({
        "context": context,
        "keywords": ", ".join(keywords)
    })

if uploaded_file:
    try:
        st.info("Extracting text from PDF...")
        raw_text = extract_text_from_pdf(uploaded_file)

        if not raw_text:
            st.warning("No text could be extracted from the PDF. It might be scanned or image-based.")
        else:
            st.success("Text extraction complete.")
            st.subheader("📚 Extracted Text")
            st.text_area("Raw Text", raw_text[:3000] + "..." if len(raw_text) > 3000 else raw_text, height=300)

            if st.button("Summarize and Extract Keywords"):
                with st.spinner("Processing with Gemini..."):
                    chunks = chunk_text(raw_text, max_tokens=800)
                    docs = [Document(page_content=chunk) for chunk in chunks]

                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

                    # Generate summary
                    summary_result = summarize_text_with_prompt(llm, docs)
                    summary_text = summary_result["text"] if isinstance(summary_result, dict) else str(summary_result)

                    # Extract raw keywords
                    raw_keywords = extract_keywords(raw_text)

                    # Refine and explain keywords
                    refined_result = refine_and_explain_keywords(llm, raw_keywords, raw_text)
                    refined_text = refined_result["text"] if isinstance(refined_result, dict) else str(refined_result)

                    # Display results
                    st.subheader("📝 Summary")
                    st.write(summary_text)

                    st.subheader("🔑 Refined Keywords & Explanations")
                    st.write(refined_text)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
