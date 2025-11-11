#          -> characters (not check word is complete or not)
# chunks |
#          -> tokens

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loaders = PyPDFLoader('./document_loader/dl-curriculum.pdf')

docs = loaders.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0, # overlapping reason of characters (retains the chunks to avoid loose context on embeddings) => 10 - 20 % is good
    separator=''
)

result = splitter.split_documents(docs)

print(len(result))
print(result[0].page_content)
print(result[0].metadata)