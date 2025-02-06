from langchain_community.document_loaders import DirectoryLoader
import os

def main():
    """This is the basic example of a loader"""
    print(get_directory_loader_sample())


def get_directory_loader_sample():
    path = os.path.join( "data", "working")
    print(path)
    return get_context(path, "**/*.md")

def get_context(directory,glob):
    loader = DirectoryLoader(directory, glob=glob, use_multithreading=True)
    docs = loader.load()
    return docs


if __name__ == "__main__":
    main()