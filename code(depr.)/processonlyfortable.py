import os
import requests
from urllib.parse import urlparse
import re
import time
from typing import List, Tuple, Optional, Dict
import logging
from IPython.display import HTML, display
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# import fitz  # PyMuPDF
import shapely.geometry as sg
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity
import concurrent.futures
from GeneralAgent import Agent
from paddleocr import PaddleOCR, draw_ocr
import ollama

model = 'gpt-4o-mini'
role_prompt = """
"""
local_prompt = """
"""
rec_prompt = """
"""
def preprocess(
        # pymupdf_content: str,
        pymupdf_table_list: List[Dict],
        image_path: List[str],
        rects:List[List[Tuple]], #座標位置
        pdf_path:str,
        openai_api_key:str,
        output_dir:str,
        ocr_lang :str='en',
) -> str:
    result=""
    ocr = PaddleOCR(use_angle_cls=True, lang=ocr_lang)  #chinese_cht
    role_prompt = """你是一個圖片摘要生成的機器人，請你對這張圖片進行摘要，若有語言出現的話請辨別該語言並輸出成文字．
"""
    local_prompt = """你是一個圖片摘要生成的機器人，請你對這張圖片進行摘要，若有語言出現的話請辨別該語言並輸出成文字．
"""
    rec_prompt = """
"""
    image_dict = dict()
    if not os.path.isdir("parse_txt"):
        os.makedirs("parse_txt")
    file_name = os.path.basename(pdf_path)
    final_output_path = "parse_txt/"+str(file_name.rstrip(".pdf"))+"_parse.txt"
    print(final_output_path)

    # 確保 parse_txt 目錄存在
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

    # 以寫入模式打開文件，這會清空文件（如果已存在）或創建新文件
    with open(final_output_path, "w") as file:
        pass  # 不寫入任何內容，只是創建或清空文件

    # 先把資料夾裡面過往跑過的資料清好，包含圖片和文字檔，再進行操作

    GPT_FLAG = True
    GPT_COUNT = 0
    NOTSAMELENGTH = 0
    def download_image(image_url, save_directory=output_dir):
        # Create the save directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Get the filename from the URL
        filename = os.path.basename(urlparse(image_url).path)
        
        # Full path to save the image
        save_path = os.path.join(save_directory, filename)
        
        # Download the image
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {filename}")
        return filename

    def find_pic_images(directory = output_dir,page_number=0):
        pic_images = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.startswith(str(page_number)+'_') and file.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    pic_images.append(os.path.join(root, file))
        return pic_images

    def filter_table(content:str)-> str:
        table_pattern = re.compile(
        r'(\|(?:[^\n]*\|)+\n'   # 匹配表格头部行
        r'\|(?:\s*[-:]+\s*\|)+\s*\n'  # 匹配表格分隔行
        r'(?:\|(?:[^\n]*\|)\n)+)'  # 匹配表格内容行
        )
        result = table_pattern.findall(content)
        return result
    
    def init_chroma(embedding_function,output_dir):
        return 0
    #utilize image and table 
    for index,i in enumerate(pymupdf_table_list):
        if GPT_FLAG == True:
            time.sleep(1.5)
            GPT_FLAG = False
        ocr_table_list = []
        final_content_list = []
        final_image_list = []
        table_markdown = []
        print(f"image:{i['images']}")
        print(f"table:{i['tables']}")
        if i['tables'] !=[] and i['images']!= []: #圖片表格都存在 ex. i[image] = 4
            #比對座標來判斷是否為表格
            table_markdown = filter_table(i['text'])
            image_list = [filename for filename in image_path if filename.startswith(str(index)+'_')] # 2個
            table_list = []
            rect = rects[index]  #取出gptpdf座標
            for j in i['tables']: # i table是pymupdf的東西
                for ind,k in enumerate(rect):
                    contract = (j['bbox'][1] - k[1]) + (j['bbox'][0]-k[0])
                    #rect_h_l = (k[2]-k[0])+(k[3]-k[1])
                    #pymu_h_l = (j['bbox'][2]-j['bbox'][0])+(j['bbox'][3]-j['bbox'][1])
                    #if (abs(rect_h_l-pymu_h_l)) < 30 and (abs(k[0]-j[0]))<30:
                    #如果gptpdf抓到的圖片位置足夠接近表格的話這張圖片就是表格
                    if (abs(contract) < 40):
                        path = str(index) + '_' + str(ind) +'.png'
                        image_list.remove(path) #2
                        table_list.append(path)
            for inde,m in enumerate(table_list):
                if inde >= len(table_markdown):
                    NOTSAMELENGTH += 1
                    final_content_list.append("NaN")
                    break

                table_role_prompt= """
                你現在是一位專注於製作HTML表格的工程師，你的任務是要畫出一個可以顯示的表格。
                """
                table_local_prompt = """
                你現在是一個製作HTML表格的工程師，你的任務是將圖片中表格的架構以及markdown的內容作結合，你必須要做到:
                1.請使用合併儲存格完整的表現出表格的結構，請勿必要遵守        
                2.你的輸出務必是完整的HTML格式，請不要省略輸出。
                3.請適度的參考Markdown的內容。
                4.請注意會有無線表格的狀況，務必使得其結構更準確。
                5. 請不要輸出除了html格式以外的任何內容，例如「以下是輸出的檔案內容」等等
                6. 使用表格內的語言回答問題。
                最後再強調一次，請確保表格的完整性與是否與圖片中的格式相符，絕對不能有跑版的狀況出現。

                """
                agent = Agent(role=table_role_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=False)
                local_prompt = table_local_prompt +table_markdown[inde]
                content = agent.run([local_prompt, {'image': output_dir+m}])
                
                # res = ollama.chat(
                #     model="llava-llama3",
                #     messages=[
                #         {
                #             'role': 'system',
                #             'content': table_role_prompt,
                #         },
                #         {
                #             'role': 'user',
                #             'content': table_local_prompt +table_markdown[inde],
                #             'images': [output_dir+m]
                #         }
                #     ]
                # )

                # print(res['message']['content'])
                
                final_content_list.append(content)
                GPT_FLAG = True
                GPT_COUNT += 1
            for inde_n,n in enumerate(image_list):

                role_prompt = """你是一個圖片摘要生成的機器人，請你對這張圖片進行摘要，若有語言出現的話請辨別該語言並輸出成文字．
                """
                local_prompt = """你是一個圖片摘要生成的機器人，請你對這張圖片進行摘要，若有語言出現的話請辨別該語言並輸出成文字．
                """
                agent = Agent(role=role_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=False)
                local_prompt = local_prompt
                content = agent.run([local_prompt, {'image': output_dir+n}])
                # res = ollama.chat(
                #     model="llava-llama3",
                #     messages=[
                #         {
                #             'role': 'system',
                #             'content': role_prompt,
                #         },
                #         {
                #             'role': 'user',
                #             'content': local_prompt ,
                #             'images': [output_dir+n]
                #         }
                #     ]
                # )
                # content = res['message']['content']
                image_dict[content] = f"{output_dir}{n}"
                print(f"response:{content}")
                final_image_list.append(content)
                GPT_FLAG = True
                GPT_COUNT += 1
        elif i['tables'] == [] and i['images']!= []: #只有圖片
            #看有幾張圖片
            image_list = [filename for filename in image_path if filename.startswith(str(index)+'_')]
            for j in image_list:
                role_prompt = """你是一個圖片摘要生成的機器人，請你對這張圖片進行摘要，若有語言出現的話請辨別該語言並輸出成文字．
                """
                local_prompt = """你是一個圖片摘要生成的機器人，請你對這張圖片進行摘要，若有語言出現的話請辨別該語言並輸出成文字．
                """
               #call llm
                agent = Agent(role=role_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=False)
                local_prompt = local_prompt
                content = agent.run([local_prompt, {'image':output_dir+j}])
                # res = ollama.chat(
                #     model="llava-llama3",
                #     messages=[
                #         {
                #             'role': 'system',
                #             'content': role_prompt,
                #         },
                #         {
                #             'role': 'user',
                #             'content': local_prompt ,
                #             'images': [output_dir+j]
                #         }
                #     ]
                # )
                # content = res['message']['content']
                image_dict[content] = output_dir + j
                print(f"response:{content}")
                final_image_list.append(content)
                GPT_FLAG = True
                GPT_COUNT += 1
        elif i['tables'] != [] and i['images'] == []: #只有表格
            #看有幾個表格 
            table_role_prompt= """
            你現在是一位專注於製作HTML表格的工程師，你的任務是要畫出一個可以顯示的表格。
            """
            table_local_prompt = """
            你現在是一個製作HTML表格的工程師，你的任務是將圖片中表格的架構以及markdown的內容作結合，你必須要做到:
                            1.請使用合併儲存格完整的表現出表格的結構，請勿必要遵守               
            2.你的輸出務必是完整的HTML格式，請不要省略輸出。
            3.請適度的參考Markdown的內容。
            4.請注意會有無線表格的狀況，務必使得其結構更準確。
            5.請不要輸出除了html格式以外的任何內容，例如「以下是輸出的檔案內容」，或是「```」。
            6.請你以圖片表格中的結構為主，參考Markdown的內容。
            最後再強調一次，請確保表格的完整性與是否與圖片中的格式相符，絕對不能有跑版的狀況出現。      
            """
            table_list = [filename for filename in image_path if filename.startswith(str(index)+'_')]
            print(f"table list:{table_list}")
            table_markdown = filter_table(i['text'])
            # for j in table_markdown:  #將純文本表格去除
            #     pymupdf_content = pymupdf_content.replace(j,'')
            for index_no_use,k in enumerate(table_list):
                if index_no_use >= len(table_markdown):
                    NOTSAMELENGTH += 1
                    final_content_list.append("NaN")
                    break
                print("table:",k)
                #call llm
                agent = Agent(role=table_role_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=False)
                local_prompt = table_local_prompt + table_markdown[index_no_use]
                content = agent.run([local_prompt, {'image': output_dir+k}])
                # res = ollama.chat(
                #     model="llava-llama3",
                #     messages=[
                #         {
                #             'role': 'system',
                #             'content': table_role_prompt,
                #         },
                #         {
                #             'role': 'user',
                #             'content': table_local_prompt + table_markdown[index_no_use] ,
                #             'images': [output_dir+k]
                #         }
                #     ]
                # )
                # content = res['message']['content']
                print(f"response:{content}")
                final_content_list.append(content)
                GPT_FLAG = True
                GPT_COUNT += 1
        elif i['tables'] == [] and i['images'] == []: #代表沒有找到表格但是說不定會有擷取到表格圖片，用OCR解
            # 如果pymupdf4llm沒有找到任何東西的話，就把資訊丟到最下面當補充，ＯＣＲ不能夠確定東西在哪裡，檔案另外存
            res = find_pic_images(page_number=index)
            print(f"res:{res}")
            if res != []:
                for image_ in res:

                    ocr_result = ocr.ocr(image_, cls=True)
                    first_chunk = ocr_result[0][0][1][0] 
                    last_chunk = ocr_result[0][-1][1][0]
                    fisrt_find = pymupdf_table_list[index]['text'].find(first_chunk[:20])
                    last_find = pymupdf_table_list[index]['text'].find(last_chunk[len(last_chunk)-20:])
                    if last_find != -1 and fisrt_find != -1:  #find the text
                        table_role_prompt= """
                        你現在是一位專注於製作HTML表格的工程師，你的任務是要畫出一個可以顯示的表格。
                        """
                        table_local_prompt = """
                        你現在是一個製作HTML表格的工程師，你的任務是將圖片中表格的架構以及markdown的表格中的內容作結合，你必須要做到:
                        1.請判斷該圖片中的上下文與表格之間是否有關聯，如果無關，可以請你無視。
                        2.請專注在圖片表格的結構，完整的表現出原本的架構。
                        3.請使用Markdown中的文字來填入表格中。
                        4.請注意合併儲存格，讓結構完整。   
                        5.請你注意表格生成的合理性，並判斷表格位置是否正確。 
                        6. 請不要輸出除了html格式以外的任何內容，例如「以下是輸出的檔案內容」等等           
                        """
                        text = pymupdf_table_list[index]['text'][fisrt_find:last_find+20]
                       # print(text)
                        agent = Agent(role=table_role_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=False)
                        local_prompt = table_local_prompt + text
                        content = agent.run([local_prompt, {'image': image_}])
                        # res = ollama.chat(
                        #     model="llava-llama3",
                        #     messages=[
                        #         {
                        #             'role': 'system',
                        #             'content': table_role_prompt,
                        #         },
                        #         {
                        #             'role': 'user',
                        #             'content': table_local_prompt + text ,
                        #             'images': [image_]
                        #         }
                        #     ]
                        # )
                        # content = res['message']['content']
                        GPT_FLAG = True
                        GPT_COUNT += 1
                        #這裡不能加東西這裡不能加東西這裡不能加東西
                        #因為是ＯＣＲ的所以一定要放在最下面
                        ocr_table_list.append(content)
                        # final_content_list.append(content)                        
                    else:
                        #當作圖片處理
                        role_prompt = """你現在是一位專注於製作HTML表格的工程師，你的任務是要畫出一個可以顯示的表格。
                         """
                        local_prompt = """你現在是一個製作HTML表格的工程師，你的任務是將圖片中表格的架構與文字轉成HTML表格，你必須要做到:
                        1.請判斷該圖片中的上下文與表格之間是否有關聯，如果無關，可以請你無視。
                        2.請專注在圖片表格的結構，完整的表現出原本的架構。
                        3.請專注在圖片文字，盡量預測正確。
                        4.請注意合併儲存格，讓結構完整。   
                        5.請你注意表格生成的合理性，並判斷表格位置是否正確。
                        6. 請不要輸出除了html格式以外的任何內容，例如「以下是輸出的檔案內容」等等
                        """
                        agent = Agent(role=role_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=False)
                        local_prompt = local_prompt
                        content = agent.run([local_prompt, {'image':image_}])
                        # res = ollama.chat(
                        #     model="llava-llama3",
                        #     messages=[
                        #         {
                        #             'role': 'system',
                        #             'content': role_prompt,
                        #         },
                        #         {
                        #             'role': 'user',
                        #             'content': local_prompt ,
                        #             'images': [image_]
                        #         }
                        #     ]
                        # )
                        # content = res['message']['content']
                        #img_filename = download_image(image_)
                        image_dict[content] = image_
                        GPT_FLAG = True
                        GPT_COUNT += 1
                        final_image_list.append(content)
        print(f"============== processing page {index} ==============")
        page_content = pymupdf_table_list[index]['text']
        with open(final_output_path, "a") as file:
            
            if len(final_content_list) > 0:
                print("Processing table...")
                print(f"length of final_content_list:{len(final_content_list)}")
                print(f"length of table_markdown:{len(table_markdown)}")
                #should be the same
                if len(final_content_list) == len(table_markdown):
                    for num_table, raw_table in enumerate(table_markdown):

                        if final_content_list[num_table] == "NaN":
                            file.write(page_content)
                            continue

                        new_page_content = page_content.replace(raw_table, final_content_list[num_table])
                        if new_page_content == page_content:
                            # 如果replace没有成功，新增内容到page_content的最下面
                            page_content += '\n' + final_content_list[num_table]
                            file.write(page_content)
                        else:
                            # 如果replace成功，更新page_content
                            page_content = new_page_content
                            file.write(page_content)
                else:
                    NOTSAMELENGTH += 1
                    for table in final_content_list:
                        file.write(f"\nsingle table: \n")
                        file.write(f"{table}")
            if len(ocr_table_list) > 0:
                print("Processing ocr table...")
                for ocr_table in ocr_table_list:
                    file.write(f"\nocr table: \n")
                    file.write(f"{ocr_table}")
            index_image = 1 
            if len(final_image_list) > 0:
                print("Processing image...")
                for image_content in final_image_list:

                    file.write(f"\nimage {index_image}: \n")
                    file.write(f"{image_content}")  #將HTML存進去txt
                    file.write(f"\nend of image {index_image}: \n")
                    index_image += 1
            if len(final_image_list)==0 and len(final_content_list)==0:
                print("No table or image found.")
                file.write(page_content)
        print(f"============== End of processing page {index} ==============")
    return final_output_path, image_dict, GPT_COUNT, NOTSAMELENGTH#, data