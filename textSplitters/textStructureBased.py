# making chunks on => para (\n\n) -> line (\n) -> words -> character


from langchain.text_splitter import RecursiveCharacterTextSplitter

spiltter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
   
)

text = '''
Technology has become an integral part of our daily lives, transforming the way we communicate, work, and learn. From smartphones to artificial intelligence, innovations continue to make tasks faster and more efficient. The internet connects people across the globe, enabling instant sharing of ideas and knowledge. However, this rapid advancement also raises concerns about privacy and job automation. Balancing progress with responsibility is crucial for a sustainable future. Ultimately, technology should serve humanity, not control it.
'''
result = spiltter.split_text(text)

print (result)