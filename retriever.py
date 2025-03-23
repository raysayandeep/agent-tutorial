from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(model='nomic-embed-text')
# pdfloader = PyPDFLoader(file_path='genai-coding-task-overview.pdf')
# pdf_document = pdfloader.load()
# splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50,separators=["\n"])
# splitted_docs = splitter.split_documents(pdf_document)

def loadPdf(filepath):
    pdfloader = PyPDFLoader(file_path=filepath)
    pdf_document = pdfloader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50,separators=["\n"])
    splitted_docs = splitter.split_documents(pdf_document)
    return splitted_docs

def faissVectorStore(embedding,documents):
    vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embedding,
    )
    return vectorstore


def saveFaissVectorStore(vector_store,filename):
    if vector_store:
        try:
            vector_store.save_local(filename+"dbindex")
            return True
        except Exception:
            return False
        
def loadFaissVectorStore(filename,embedding):
    dbname = filename+"dbindex"
    try:
        vector_store = FAISS.load_local(dbname, embedding, allow_dangerous_deserialization=True)
        return vector_store
    except Exception:
        return False

def loadAndSave(filepath):
    documents = loadPdf(filepath)
    vector_store  = faissVectorStore(embeddings,documents)
    status = saveFaissVectorStore(vector_store,filepath)
    return status

def retrieveDocument(question:str):
    #question = state["question"]
    filepath="genai-coding-task-overview.pdf"
    vector_store = loadFaissVectorStore(filepath,embeddings)
    documents = vector_store.similarity_search_with_score(question,k=4)
    #return {"documents": documents, "question": question}
    #for doc in documents:
        #print(doc[0].page_content)
    #    print(type(doc[0]))
    return "\n\n".join([doc[0].page_content for doc in documents]) if documents else "No relevant documents found."

if __name__ == "__main__":

    # filepath="genai-coding-task-overview.pdf"
    # #print(loadAndSave(filepath))
    # vector_store = loadFaissVectorStore(filepath,embeddings)


    # #retriever = vstore.as_retriever(search_kwargs={"k": 10})
    # #print(retriever.invoke("What is the title"))

    # query="What is Gen AI"
    # emb_query = embeddings.embed_query(query)
    # sub_docs = vector_store.similarity_search_with_score(query,k=2)
    # sub_docs1 = vector_store.similarity_search_with_score_by_vector(emb_query,k=2)
    # print(sub_docs)
    # print(sub_docs1)

    
    print(retrieveDocument("What is Gen AI"))