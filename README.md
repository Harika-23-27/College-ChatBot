# GenAI College Chatbot

A simple Generative AI-based College Chatbot that answers questions using information from college PDF documents.

## Features

* Answers questions based on college documents
* Reads information from PDF files
* Uses FAISS for storing and searching document information
* Uses Hugging Face embeddings for semantic search
* Uses FLAN-T5 to generate answers
* Simple user interface using Streamlit

## Technologies Used

* Python
* Streamlit
* Hugging Face Transformers
* Sentence Transformers
* LangChain
* FAISS
* PyPDF

## How It Works

1. PDF documents are added to the `data` folder.
2. The text is extracted from the PDFs.
3. The text is divided into smaller chunks.
4. The chunks are converted into embeddings.
5. FAISS stores the embeddings and performs similarity search.
6. Relevant information is given to the FLAN-T5 model.
7. The chatbot generates an answer based on the retrieved information.

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The chatbot will open in your browser.

## Example Questions

You can ask questions such as:

* What courses are available?
* What is the fee structure?
* What is the syllabus for a particular course?
* What are the college facilities?

## Future Improvements

* Add more college documents
* Improve answer accuracy
* Add chat history
* Deploy the chatbot online
* Improve the user interface

## Author

**Harika Veduru**
