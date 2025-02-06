from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from loaders.WebBaseLoader import get_web_base_loader_sample, get_context 
from loaders.DirectoryLoader import get_directory_loader_sample
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

PROMPT_TEMPLATE = """
Answer the question based mostly on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

PROMPT_TEMPLATE_BP = """
Answer the question based on the following context:
{context}
---
Answer the question based on the above context: {question}
"""


def main():
    data = get_directory_loader_sample()
    vector_store = populate_information(data)
    while (prompt := input("Enter a prompt (q to quit): ")) != "q":
        result = query_rag(prompt, vector_store)
        print(result)


def populate_information(documents):
    chunks = split_documents(documents)
    return vector_store( chunks )
    
    
def query_rag(query_text: str, vector_store):
    result = vector_store.similarity_search(query=query_text,k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in result])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = OllamaLLM(model="llama3")
    response_text = model.invoke(prompt)
    return response_text

def vector_store( documents):
    vector_store = InMemoryVectorStore(get_embedding_function())
    vector_store.add_documents(documents=documents)
    return vector_store


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(documents)
    return texts


if __name__ == "__main__":
    main()
