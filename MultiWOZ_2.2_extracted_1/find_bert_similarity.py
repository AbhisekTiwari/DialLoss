from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('bert-base-nli-mean-tokens')

def cosine(embedding1,embedding2):
    return cosine_similarity([embedding1],[embedding2])[0][0]

texts=[]
with open('train_trg.txt','r') as reader:
    for line in reader:
        texts.append(line)

result=open('result1.txt','w')

for text in texts[:30]:
    embedding1=model.encode(text)
    current_text_results=[]
    result.write(text+'\n\n')
    print(text)
    for text1 in texts:
        embedding2=model.encode(text1)
        bert_score=cosine(embedding1,embedding2)
        current_text_results.append([bert_score,text1])

    current_text_results=sorted(current_text_results, key=lambda x: x[0], reverse=True)[:100]
    
    for value in current_text_results:
        result.write(f'{value[0]}, {value[1]}\n')
    