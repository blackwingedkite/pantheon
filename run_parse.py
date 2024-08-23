from parse import parse_pdf
from processonlyfortable_hit import preprocess
import pymupdf4llm
import re
import os
import warnings
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI


def run_parse(pdf_path, openai_api_key,finance,page):
    print("start processing: ", pdf_path)
    image_paths,recs, output_dir = parse_pdf(pdf_path,finance=finance,minimun_merge_size=40,merge_distance=60,horizontal_merge_distance=60,near_distance=5,horizontal_near_distance=5,page = page)
    print(image_paths)
    print(f"start using pymupdf4llm to markdown: {pdf_path}")
    find_table = pymupdf4llm.to_markdown(pdf_path,page_chunks=True, write_images=False,margins=(0,0,0,0))
    print(f"total image paths:{image_paths}")
    print("====================")
    print(output_dir)
    print("====================")
    final_txt_path, image_description_list, GPT_CALL_COUNT, NOTSAMELENGTH = preprocess(find_table,image_paths,recs,pdf_path,openai_api_key=openai_api_key,output_dir=output_dir, ocr_lang="en") #ocr_lang = chinese_cht or en ; md_text,
    print("end of processing: ", pdf_path)
    return final_txt_path, image_description_list, GPT_CALL_COUNT, NOTSAMELENGTH


#   chunking 方式1:非html部分400, 200，非chunking不重複
def chunk_content(text, chunk_size=400, overlap=200):
    chunks = []
    
    patterns = [
        (r'<table.*?>.*?</table>', 'table'),
        (r'image \d+:.*?end of image \d+:', 'image'),
        (r'<(div|p|h[1-6]).*?>.*?</\1>', 'html')
    ]
    
    matches = []
    
    for pattern, type in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            matches.append((match.start(), match.end(), type, match.group()))
    
    matches.sort(key=lambda x: x[0])
    
    def split_text(text, chunk_size, overlap):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    last_end = 0
    for start, end, type, content in matches:
        if start > last_end:
            text_chunk = text[last_end:start].strip()
            if text_chunk:
                text_chunks = split_text(text_chunk, chunk_size, overlap)
                chunks.extend(text_chunks)
        
        chunks.append(content)
        
        last_end = end
    
    if last_end < len(text):
        text_chunk = text[last_end:].strip()
        if text_chunk:
            text_chunks = split_text(text_chunk, chunk_size, overlap)
            chunks.extend(text_chunks)
    
    return chunks

# chunking方式2:我希望可以稍微改變策略：改成chunk_size為200 overlap為０，但是在輸出chunking的時候把兩個chunk合而為一，我指的是1,2組合成一組，2,3組合成一組，3,4組合成一組，以此類推，html檔案也做同樣的處理
#目前還沒實現：image 和end of image之間不要被合併
def merge_chunk_content(text, chunk_size=200):
    chunks = []
    
    patterns = [
        (r'<table.*?>.*?</table>', 'table'),
        (r'image \d+:.*?end of image \d+:', 'image'),
        (r'<(div|p|h[1-6]).*?>.*?</\1>', 'html')
    ]
    
    matches = []
    
    for pattern, type in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            matches.append((match.start(), match.end(), type, match.group()))
    
    matches.sort(key=lambda x: x[0])
    
    def split_text(text, chunk_size):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    last_end = 0
    for start, end, type, content in matches:
        if start > last_end:
            text_chunk = text[last_end:start].strip()
            if text_chunk:
                text_chunks = split_text(text_chunk, chunk_size)
                chunks.extend(text_chunks)
        
        html_chunks = split_text(content, chunk_size)
        chunks.extend(html_chunks)
        
        last_end = end
    
    if last_end < len(text):
        text_chunk = text[last_end:].strip()
        if text_chunk:
            text_chunks = split_text(text_chunk, chunk_size)
            chunks.extend(text_chunks)

    # 合併相鄰的chunks
    merged_chunks = []
    for i in range(len(chunks) - 1):
        if ("end of image" in chunks[i]) and ("end of image" not in chunks[i+1]):
            merged_chunks.append(chunks[i])
            merged_chunks.append(chunks[i+1])
        elif ("end of image" in chunks[i]) and ("end of image" in chunks[i+1]):
            merged_chunks.append(chunks[i])
        else:
            merged_chunks.append(chunks[i] + "\n" + chunks[i+1])

        if i==(len(chunks)-2) and ("end of image" in chunks[i+1]):
            merged_chunks.append(chunks[i+1])    
    return merged_chunks


# 儲存分塊到檔案
def save_chunks_to_file(chunks, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n')
            f.write('\n=============\n')

def add_chunk_to_db(chunks, collection):
    # 將 chunks 添加到 Chroma 集合中
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            metadatas=[{"source": f"chunk_{i}"}],
            ids=[f"id_{i}"]
        )


def query_and_respond(query, openai_api_key ,collection, k=3):
    # 從 Chroma 檢索相關文檔
    api_key = openai_api_key
    client = OpenAI()
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    # 準備 prompt
    context = "\n".join(results['documents'][0])
    prompt = f"""Given the following context and question, please provide a relevant answer. If the context doesn't contain enough information to answer the question, please say so.
                Context: {context}
                Question: {query}
                Answer:"""

    # 使用 GPT 生成回答
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def query_and_respond(query ,collection, k=3): #openai_api_key
    # 從 Chroma 檢索相關文檔
    # api_key = openai_api_key
    client = OpenAI()
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    # 準備 prompt
    context = "\n".join(results['documents'][0])

    prompt = f"""Given the following context and question, please provide a relevant answer. If the context doesn't contain enough information to answer the question, please say so.
                Context: {context}
                Question: {query}
                Answer:"""

    # 使用 GPT 生成回答
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def query_only(query ,collection, k=3): #openai_api_key
    # 從 Chroma 檢索相關文檔
    # api_key = openai_api_key
    client = OpenAI()
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    # 準備 prompt
    context = "\n".join(results['documents'][0])
    return context

# # 使用範例
# query = "請問A+B+C+D 合計要幾學分?"
# answer = query_and_respond(query)
# print(f"Question: {query}")
# print(f"Answer: {answer}")

#備份
# print(type(pdf_path)) <class 'str'>
# print(type(image_paths)) <class 'list'>
# print(type(recs)) <class 'list'>
# print(type(md_text)) <class 'str'>
# print(type(find_table)) <class 'list'>

# from parse import parse_pdf
# from parse import plt_img_base64
# from parse import _parse_pdf_to_images
# import pymupdf4llm
# #import google.generativeai as genai
# import chromadb
# #from sentence_transformers import SentenceTransformer
# import getpass
# import base64
# import httpx
# #from langchain_core.messages import HumanMessage
# import os
# import fnmatch
# from GeneralAgent import Agent
# from langchain_openai import ChatOpenAI
# #from process import preprocess
# from processonlyfortable import preprocess
# import warnings
# warnings.filterwarnings("ignore")
# pdf_path = "image+word.pdf"
        

# def run_parse(pdf_path):
#     image_paths,recs = parse_pdf(pdf_path)
#     # md_text = pymupdf4llm.to_markdown(pdf_path)
#     find_table = pymupdf4llm.to_markdown(pdf_path,page_chunks=True, write_images=False)
#     # print(image_paths) # ['2_0.png', '3_0.png', '4_0.png']
#     # print(recs) # [[], [], [(72.0, 71.300048828125, 679.06005859375, 306.30999755859375)], [(75.83999633789062, 31.4400634765625, 740.3799438476562, 486.2200012207031)], [(75.83999633789062, 31.44000244140625, 770.0199584960938, 421.510009765625)], [], []]
#     # print(find_table) # dict, 有metadata, md的table
#     # file_paths = []
#     # folder_path = "./"
#     # # 遍歷資料夾中的所有檔案
#     # for root, dirs, files in os.walk(folder_path):
#     #     for file in files:
#     #         # 檢查檔案名稱是否包含 "test" 並且副檔名是 ".png"
#     #         if fnmatch.fnmatch(file, f'*{pdf_name}*.png'):
#     #             # 如果符合條件，將完整的檔案路徑加入列表
#     #             file_paths.append(os.path.join(root, file))
#     final_pdf_path, image_description_list = preprocess(find_table,image_paths,recs,pdf_path,None) #md_text,
#     print(final_pdf_path)
#     return 0 #final_pdf_path

# if __name__ == '__main__':
#     run_parse(pdf_path)





