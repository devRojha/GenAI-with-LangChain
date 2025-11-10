from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path= './document_loader/Books',
    glob= '*.pdf',
    loader_cls= PyPDFLoader
)

docs = loader.load()
docs = loader.lazy_load() # load docs one by one

# print(len(docs))

print (docs[45].page_content)
print (docs[1].metadata)