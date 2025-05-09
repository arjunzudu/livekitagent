import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 1: Load and split the PDF from data folder
pdf_path = os.path.join("data", "document.pdf")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True
)
texts = text_splitter.split_documents(documents)

# Step 2: Create or load FAISS vector store
embeddings = OpenAIEmbeddings()
faiss_index_path = "faiss_index"  # Directory to store FAISS index

# Check if FAISS index exists on disk
if os.path.exists(faiss_index_path):
    print("Loading existing FAISS index from disk...")
    vectorstore = FAISS.load_local(
        faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True  # Required for loading pickled files
    )
else:
    print("Creating new FAISS index...")
    vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)
    # Save the index to disk
    vectorstore.save_local(faiss_index_path)
    print(f"FAISS index saved to {faiss_index_path}")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Step 3: Set up the RAG pipeline
llm = ChatOpenAI(model="gpt-4o", temperature=0)

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer short and precise.

{context}

Question: {question}

Answer: """
custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# Step 4: CLI Chatbot Loop
def main():
    print("Welcome to the PDF Q&A Chatbot! Type 'exit' to quit.")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        try:
            answer = rag_chain.invoke(question)
            print("\nAnswer:", answer)
        except Exception as e:
            print("\nError:", str(e))

if __name__ == "__main__":
    main()