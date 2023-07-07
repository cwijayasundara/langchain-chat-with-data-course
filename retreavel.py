import os
import openai
import sys

sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print('There are ', vectordb._collection.count(), 'documents in the vectorstore')


# Addressing Diversity: Maximum marginal relevance
question = "what did they say about matlab?"
docs_ss = vectordb.similarity_search(question, k=3)
print(len(docs_ss))

if len(docs_ss) > 0:
    print(docs_ss[0].page_content[:100])
    print(docs_ss[1].page_content[:100])

docs_mmr = vectordb.max_marginal_relevance_search(question, k=3)
if len(docs_mmr) > 0:
    print(docs_mmr[0].page_content[:100])
    print(docs_mmr[1].page_content[:100])

# Addressing Specificity: working with metadata

question = "what did they say about regression in the third lecture?"
docs = vectordb.similarity_search(
    question,
    k=3,
    filter={"source": "docs/cs229_lectures/MachineLearning-Lecture03.pdf"}
)
for d in docs:
    print(d.metadata)

# Addressing Specificity: working with metadata using self-query retriever
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of "
                    "`docs/cs229_lectures/machinelearning-lecture01.pdf`, "
                    "`docs/cs229_lectures/machinelearninglLecture02.pdf`, "
                    "or `docs/cs229_lectures/machinelearning-lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]

document_content_description = "Lecture notes"
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)
question = "what did they say about regression in the third lecture?"
docs = retriever.get_relevant_documents(question)
for d in docs:
    print(d.metadata)

# Additional tricks: compression
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


# Wrap our vectorstore
llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)
question = "what did they say about matlab?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)

#  combining compression and self-query
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type="mmr")
)
question = "what did they say about matlab?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)

# Other types of retrieval

from langchain.retrievers import SVMRetriever
from langchain.retrievers import TFIDFRetriever
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
all_page_text = [p.page_content for p in pages]
joined_page_text = " ".join(all_page_text)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_text(joined_page_text)

# Retrieve
svm_retriever = SVMRetriever.from_texts(splits, embedding)
tfidf_retriever = TFIDFRetriever.from_texts(splits)
question = "What are major topics for this class?"
docs_svm = svm_retriever.get_relevant_documents(question)
docs_svm[0]
question = "what did they say about matlab?"
docs_tfidf = tfidf_retriever.get_relevant_documents(question)
docs_tfidf[0]
