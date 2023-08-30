import flask
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline
import string
from tqdm.autonotebook import tqdm
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util


tokenizer = AutoTokenizer.from_pretrained("hogger32/xlmRoberta-for-VietnameseQA")
# model = AutoModelForQuestionAnswering.from_pretrained("hogger32/xlmRoberta-for-VietnameseQA")
qa_model = pipeline("question-answering", model="hogger32/xlmRoberta-for-VietnameseQA")


my_data = pd.read_excel("final_chatbot_dataset.xlsx")

my_contexts = my_data["Context"].tolist()
# print(my_contexts)
stopword_file = open('vietnamese-stopwords-dash.txt', 'r')
stopword = stopword_file.read()
stopword = stopword.split('\n')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
my_corpus = []

def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and token not in stopword:
            tokenized_doc.append(token)
    return tokenized_doc

my_corpus = []
for context in tqdm(my_contexts):
  my_corpus.append(bm25_tokenizer(context))
my_bm25 = BM25Okapi(my_corpus)

class BM25():
    def __int__(self):
        self.qa_model = qa_model
        self.data = my_data
        self.contexts = my_contexts
        self.bm25_corpus = my_bm25

    def my_search(self, query):
        #     print("Input question:", query)

        ##### BM25 search (lexical search) #####
        bm25_scores = my_bm25.get_scores(bm25_tokenizer(query))
        # print(bm25_scores)
        top_n = np.argpartition(bm25_scores, -5)[-5:]
        bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

        # print("Top-3 lexical search (BM25) hits")
        ##### Re-Ranking #####
        # Now, score all retrieved passages with the cross_encoder
        cross_inp = [[query, my_contexts[hit['corpus_id']]] for hit in bm25_hits]
        cross_scores = cross_encoder.predict(cross_inp)

        # Sort results by the cross-encoder scores
        for idx in range(len(cross_scores)):
            bm25_hits[idx]['cross-score'] = cross_scores[idx]

        bm25_hits = sorted(bm25_hits, key=lambda x: x['cross-score'], reverse=True)

        results = []
        for hit in bm25_hits[0:5]:
            #         print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))
            results.append(my_contexts[hit['corpus_id']].replace("\n", " "))
        return results

    def answer(self, question):
        search = self.my_search(question)[0]
        res = qa_model(question=question, context=search)
        return res.answer

bm25_base_qa = BM25()

# print(bm25_base_qa.my_search(query = "Nội dung của chi phí cấu thành chi phí thuê dịch vụ theo yêu cầu là gì?")[0])






