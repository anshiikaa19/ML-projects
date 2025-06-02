import streamlit as st
from dotenv import load_dotenv
import asyncio
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import re
from docx import Document

# Load environment variables
load_dotenv()

# Define the message limit
MESSAGE_LIMIT = 10

# Streamlit application
st.title("Memory-Enabled Chatbot & Student Evaluation App")
st.write("Choose a mode to start:")

# Initialize session state for mode if not already done
if 'mode' not in st.session_state:
    st.session_state.mode = None

# Initialize session state for conversation if not already done
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()


# Function to trim message history if it exceeds the limit
def trim_history(memory, limit):
    while len(memory.chat_memory.messages) > limit:
        memory.chat_memory.messages.pop(0)


# Function to generate chat responses
async def generate_response(user_input):
    st.session_state.memory.chat_memory.messages.append(
        HumanMessage(content=user_input))

    # Display the user's message
    message(user_input, is_user=True)

    # Prepare the messages for LangChain
    past_messages = [
        SystemMessage(content="You are a helpful AI bot. Your name is Carl.")]
    for msg in st.session_state.memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            past_messages.append(HumanMessage(content=msg.content))
        else:
            past_messages.append(AIMessage(content=msg.content))

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", past_messages[0].content)] +
        [("user", msg.content) if isinstance(msg, HumanMessage) else (
        "assistant", msg.content) for msg in past_messages[1:]]
    )
    parser = StrOutputParser()
    model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
    chain = prompt_template | model | parser

    # Generate and stream the response
    response = ""
    response_container = st.empty()

    async for chunk in chain.astream({"topic": user_input}):
        response += chunk
        response_container.markdown(response)

    return response


def message(content, is_user=False):
    if is_user:
        st.write(f"You: {content}")
    else:
        st.write(f"AI: {content}")


def chat_interface():
    user_input = st.text_input("You: ", "")

    if user_input:
        if user_input.lower() == 'exit':
            st.write("Exiting the conversation.")
            return

        # Get and stream the response
        response = asyncio.run(generate_response(user_input))

        # Append the AI's response
        st.session_state.memory.chat_memory.messages.append(
            AIMessage(content=response))

        # Trim the message history if it exceeds the limit
        trim_history(st.session_state.memory, MESSAGE_LIMIT)

    # Show the conversation history in an expander
    with st.expander("Conversation History"):
        for i, msg in enumerate(st.session_state.memory.chat_memory.messages):
            if isinstance(msg, HumanMessage):
                st.markdown(f"**User:** {msg.content}")
            else:
                st.markdown(f"**AI:** {msg.content}")


def extract_text_from_image(file):
    try:
        # Open image using PIL
        image = Image.open(file)

        # Preprocess image (if needed)
        # Example: Convert to grayscale
        image = image.convert('L')

        # Use pytesseract to extract text
        text = pytesseract.image_to_string(image)

        return text

    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


async def evaluate_text_with_gpt4(text):
    evaluation_criteria = (
        "Evaluate the provided essay and provide a concise evaluation summary considering the following aspects:\n"
        "- Thesis clarity: Clear and focused thesis statement that addresses the essay prompt\n"
        "- Analysis Depth: In-depth analysis of the topic with supporting evidence\n"
        "- Organization: Well structured and logically organized essay\n"
        "- Writing Clarity: Clear and concise writing with proper grammar and punctuation\n"
        "- Conclusion\n\n"
        "Conclude the evaluation with a final rating in the format 'Overall Rating: X.X/5'."
    )

    system_message = SystemMessage(
        content=f"You are an AI language model that evaluates student essays. {evaluation_criteria}"
    )
    user_message = HumanMessage(content=text)

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_message.content), ("user", user_message.content)]
    )
    parser = StrOutputParser()
    model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
    chain = prompt_template | model | parser

    # Generate and stream the response
    response = ""
    response_container = st.empty()

    async for chunk in chain.astream({"topic": text}):
        response += chunk
        response_container.markdown(response)

    return response


def extract_overall_rating(response):
    # Updated pattern to match the format 'Overall Rating: X.X/5'
    match = re.search(r"Overall Rating: (\d+(\.\d+)?)/5", response,
                      re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def student_evaluation_interface():
    st.header("Student Evaluation App")
    st.header("Choose Subject")
    option = st.selectbox(
        "Select the subject of the document",
        ("Essay-Writing-PDF", "Essay-Writing-Handwritten", "Chemistry",
         "Maths"),
        index=None,
        placeholder="Select the subject...",
    )
    st.header("Upload your document")
    uploaded_file = st.file_uploader("Choose a file",
                                     type=["pdf", "jpg", "png", "docx"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension in ["jpg", "jpeg", "png"]:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Handwritten Essay',
                     use_column_width=True)
            text = extract_text_from_image(
                uploaded_file)  # Extract text from image

        elif file_extension == "docx":
            text = extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file format!")
            return

        # Print evaluation criteria
        evaluation_criteria = (
            "Evaluate the provided essay and provide a concise evaluation summary considering the following aspects:\n"
            "- Thesis clarity: Clear and focused thesis statement that addresses the essay prompt\n"
            "- Analysis Depth: In-depth analysis of the topic with supporting evidence\n"
            "- Organization: Well structured and logically organized essay\n"
            "- Writing Clarity: Clear and concise writing with proper grammar and punctuation\n"
            "- Conclusion\n\n"

        )
        st.subheader("Evaluation Criteria")
        st.write(evaluation_criteria)

        if st.button("Start Evaluation"):
            # Evaluate the text with GPT-4
            st.subheader("Evaluation")
            response = asyncio.run(evaluate_text_with_gpt4(text))

            # Extract and display overall rating
            overall_rating = extract_overall_rating(response)
            if overall_rating is not None:
                st.subheader("Overall Rating")
                st.write(f"Rating: {overall_rating} / 5")
            else:
                st.write("No overall rating found in the evaluation.")

            # Button to rerun evaluation
            if st.button("Rerun Evaluation"):
                st.session_state.memory.chat_memory.messages.clear()
                st.text_input("You: ", value="", key="user_input")
                st.empty()


# Buttons for mode selection
if st.button("Chat"):
    st.session_state.mode = "Chat"
elif st.button("Student Evaluation"):
    st.session_state.mode = "Student Evaluation"

# Show the appropriate interface based on the selected mode
if st.session_state.mode == "Chat":
    chat_interface()
elif st.session_state.mode == "Student Evaluation":
    student_evaluation_interface()

