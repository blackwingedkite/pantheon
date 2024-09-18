from parse import parse_pdf
from preprocess import preprocess
from chunking_new import chunking
from embedding import add_chunk_to_db
import pymupdf4llm
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import time
import matplotlib.pyplot as plt
load_dotenv()

def pdf_to_json(pdf_path, model, is_ocr, ocr_lang, finance, mac_page=False):
    start_time = time.time()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    file_name_with_ext = os.path.basename(pdf_path)
    file_name = os.path.splitext(file_name_with_ext)[0]
    image_paths, recs, output_dir = parse_pdf(pdf_path, finance=finance, minimun_merge_size=40, merge_distance=60, horizontal_merge_distance=60, near_distance=5, horizontal_near_distance=5, page=mac_page)
    parse_time = time.time()
    pymupdf_content = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, write_images=False, margins=(0,0,0,0))    
    pymupdf_time = time.time()
    txt_pages, image_dict, markdown_to_html, GPT_CALL_COUNT = preprocess(pymupdf_content, image_paths, recs, output_dir, pdf_path, openai_api_key, model, is_ocr, ocr_lang) #ocr_lang = chinese_cht or en  
    with open(f"{file_name}.json", "w", encoding="utf-8") as file:
        json.dump(txt_pages, file, ensure_ascii=False, indent=4)
    with open(f"{file_name}_md2html.json", 'w', encoding="utf-8") as f:
        json.dump(markdown_to_html, f)
    times = {
        "parse_py": parse_time - start_time,
        "pymupdf": pymupdf_time - parse_time,
        "preprocess": time.time() - pymupdf_time,
    }
    return file_name, image_dict, GPT_CALL_COUNT, times

def json_to_db(file_name, chunk_size, model, merge_tables, collection):
    with open(f"{file_name}.json", 'r') as f:
        txt_pages = json.load(f)
    with open(f"{file_name}_md2html.json", 'r') as f:
        markdown_to_html = json.load(f)
    chunking_time = time.time()
    merged_chunks_page, markdown_to_html = chunking(txt_pages_dict=txt_pages, markdown_to_gptcontent=markdown_to_html, model=model, chunk_size=chunk_size, merge_tables=merge_tables)
    db_time = time.time()
    add_chunk_to_db(merged_chunks_page, collection)
    times = {
        "chunking": db_time - chunking_time,
        "add_to_db": time.time() - db_time,
    }
    return merged_chunks_page, markdown_to_html, times

def pdf_to_db(pdf_path, chunk_size, model, merge_tables, collection, is_ocr, ocr_lang, finance, mac_page=False):
    file_name, image_dict, GPT_CALL_COUNT, pdf_to_json_times = pdf_to_json(pdf_path, model, is_ocr, ocr_lang, finance, mac_page)
    merged_chunks_page, markdown_to_html, json_to_db_times = json_to_db(file_name, chunk_size, model, merge_tables, collection)
    all_times = {**pdf_to_json_times, **json_to_db_times}
    visualize_time_consumption(all_times, file_name)
    return merged_chunks_page, markdown_to_html, image_dict



def pdf_to_db_nosave(pdf_path, chunk_size, model, merge_tables, collection, is_ocr, ocr_lang, finance, mac_page=False):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    image_paths, recs, output_dir = parse_pdf(pdf_path, finance=finance, minimun_merge_size=40, merge_distance=60, horizontal_merge_distance=60, near_distance=5, horizontal_near_distance=5, page=mac_page)
    pymupdf_content = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, write_images=False, margins=(0,0,0,0))    
    txt_pages, image_dict, markdown_to_html, GPT_CALL_COUNT = preprocess(pymupdf_content, image_paths, recs, output_dir, pdf_path, openai_api_key, model, is_ocr, ocr_lang) #ocr_lang = chinese_cht or en  
    merged_chunks_page, markdown_to_html = chunking(txt_pages_dict=txt_pages, markdown_to_gptcontent=markdown_to_html, model=model, chunk_size=chunk_size, merge_tables=merge_tables)
    add_chunk_to_db(merged_chunks_page, collection)
    return merged_chunks_page, markdown_to_html, image_dict





def visualize_time_consumption(times,name):
    name = name.replace('.pdf','.png')
    steps = list(times.keys())
    durations = list(times.values())
    plt.figure(figsize=(10, 6))
    plt.bar(steps, durations)
    plt.title('Time spent on each step')
    plt.xlabel('steps')
    plt.ylabel('time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('time_consumption.png')
    plt.close()

def query_and_respond(query ,collection, markdown_to_html,image_dict, k=3):
    # 從 Chroma 檢索相關文檔
    client = OpenAI()
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    # 处理检索到的文档，将markdown替换为html，并收集图片描述
    related_image = []
    processed_documents = []
    added_html = set()  # 使用 set 提高查找效率

    for doc in results['documents'][0]:
        # 只替换需要替换的部分
        for key, value in markdown_to_html.items():
            if key in doc and key not in added_html:
                doc = doc.replace(key, value)
                added_html.add(key)  # 使用 add 方法替换 append
        processed_documents.append(doc)
            
        # 检查文档中是否包含图片描述
        for description, path in image_dict.items():
            if description in doc:
                related_image.append(path)


    # results['documents'][0] = processed_documents
    # 準備 prompt


    context = "\n".join(processed_documents)
    prompt = f"""根據以下提供的用戶詢問以及文字敘述內容，盡可能回答用戶的問題，但請不要回答在提供的知識以外的內容．
                參考資料: {context}
                用戶詢問: {query}
                請你用繁體中文回答問題，如果提供的文本有html內容，請盡可能地閱讀該html內容並且作為回覆的重要依據
                再次強調，你的回答對我們來說非常重要，我正在盯者你．
                
                你的回答:"""

    # 使用 GPT 生成回答
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一個善解人意的AI助手.你應該要盡可能地根據提供的文本內容來回答問題"},
            {"role": "user", "content": prompt}
        ]
    ) 

    return response.choices[0].message.content, related_image

def query_only(query ,collection, markdown_to_html,image_dict, k=3): 
    client = OpenAI()
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    # 处理检索到的文档，将markdown替换为html，并收集图片描述
    processed_documents = []
    related_image = []
    for doc in results['documents'][0]:
        for key, value in markdown_to_html.items():
            if key in doc:
                doc = doc.replace(key, value)
        processed_documents.append(doc)
        
        # 检查文档中是否包含图片描述
        for description, path in image_dict.items():
            if description in doc:
                related_image.append(path)
    
    # 更新结果
    results['documents'][0] = processed_documents
    # 準備 prompt
    context = "\n".join(results['documents'][0])

    return context, related_image