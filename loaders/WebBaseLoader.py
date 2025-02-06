from langchain_community.document_loaders import WebBaseLoader

def main():
    """This is the basic example of a loader"""
    print(get_web_base_loader_sample())


def get_web_base_loader_sample():
    return get_context("https://python.langchain.com/docs/integrations/document_loaders/web_base/");

def get_context(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs


if __name__ == "__main__":
    main()