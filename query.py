from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from chains.WebBaseLoader import get_web_base_loader_sample


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    while (prompt := input("Enter a prompt (q to quit): ")) != "q":
        result = query_rag(prompt, get_web_base_loader_sample())
        print(result)


def query_rag(query_text: str, context_data):

    context_text = "\n\n---\n\n".join([doc.page_content for doc in context_data])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = OllamaLLM(model="llama3")
    response_text = model.invoke(prompt)

    return response_text


if __name__ == "__main__":
    main()
