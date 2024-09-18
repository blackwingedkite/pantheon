import os
import re
from typing import List, Tuple, Optional, Dict
import logging
import ast
from IPython.display import HTML, display
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from GeneralAgent import Agent
import fitz

def clean_string(text):
    # 移除markdown圖片, 超連結
    pattern =  r'[!]\[([^\]]+)\]\(([^\)]+)\)'
    text = re.sub(pattern, '', text)
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(pattern, '', text)
    # 移除換行符號
    text = text.replace('\n', '')
    return text


def preprocess(
        pymupdf4llm_list: List[Dict],
        image_paths: List[str],
        rects:List[List[Tuple]], #座標位置
        output_dir:str,
        pdf_path:str,
        openai_api_key:str,
        model :str = "gpt-4o-mini",
        is_ocr:bool=False,
        ocr_lang :str='chinese_cht',
) -> str:
    if is_ocr:
        from paddleocr import PaddleOCR, draw_ocr

    def filter_table(content:str)-> str:
        table_pattern = re.compile(
        r'(\|(?:[^\n]*\|)+\n'   # 匹配表格头部行
        r'\|(?:\s*[-:]+\s*\|)+\s*\n'  # 匹配表格分隔行
        r'(?:\|(?:[^\n]*\|)\n)+)'  # 匹配表格内容行
        )
        result = table_pattern.findall(content)
        return result

    def find_pic_images(directory,page_number=0):
        pic_images = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.startswith(str(page_number)+'_') and file.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    pic_images.append(os.path.join(root, file))
        return pic_images


    def check_same_table(table_path, table_markdown_text, ocr_table=False):
        agent = Agent(role=check_same_table_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=False)
        local_prompt = table_checksame_prompt + table_markdown_text
        content = agent.run([local_prompt, {'image': output_dir+table_path}])
        if "true" in content or "True" in content:
            return True
        else:
            return False


    # filtered_imagepath_to_pageloc
    def check_table_or_image(pymupdf_table_list, filtered_imagepath_to_pageloc, output_dir):
        rect＿list = [i[1] for i in filtered_imagepath_to_pageloc]
        name＿list = [i[0] for i in filtered_imagepath_to_pageloc]
        table_to_path = dict()
        sole_table_path = list()

        pymupdf_table_list_backup = list()
        # pymu_table[bbox]是座標位置
        for i in pymupdf_table_list:
            pymupdf_table_list_backup.append(i)
        def find_biggest_index(name＿list):
            largest_num = -1
            for file in name＿list:
                start = file.index('_') + 1
                end = file.index('.png')
                num = int(file[start:end])
                if num > largest_num:
                    largest_num = num
            return largest_num
        
        for pymu_table in pymupdf_table_list:
            #parse_table也是座標位置
            for table_index, parse_table in enumerate(rect＿list):
                contract = (pymu_table['bbox'][1] - parse_table[1]) + (pymu_table['bbox'][0]-parse_table[0])
                rect_h_l = (parse_table[2]-parse_table[0])+(parse_table[3]-parse_table[1])
                pymu_h_l = (pymu_table['bbox'][2]-pymu_table['bbox'][0])+(pymu_table['bbox'][3]-pymu_table['bbox'][1])
                if (abs(rect_h_l-pymu_h_l)) < 20 and (abs(parse_table[0]-pymu_table['bbox'][0])) < 10:
                # if contract < 40:
                    path = str(page_index) + '_' + str(table_index) +'.png' 
                    table_to_path[str(pymu_table)] = path
                    pymupdf_table_list_backup.remove(pymu_table)
    
        if len(pymupdf_table_list_backup) > 0:
            inner_index = find_biggest_index(name＿list)
            for inner_table in pymupdf_table_list_backup:
                inner_index += 1
                rect = fitz.Rect(inner_table['bbox'])
                pix = pdf_document[page_index].get_pixmap(clip=rect, matrix=fitz.Matrix(4, 4))
                name = f'{page_index}_{inner_index}.png'
                sole_table_path.append(name)
                pix.save(os.path.join(output_dir, name))

        return table_to_path

    def check_row_column(table):
        rows = table.strip().split('\n')
        num_rows = len(rows) - 1  # 減去頭部行和分隔行
        num_cols = len(rows[0].split('|')) - 2  # 減去開頭和結尾的空字符串
        return num_rows, num_cols



    def make_path_to_markdown(table_to_path, table_markdown, GPT_COUNT):
        #現在我們有的東西：
        #1. 表格所有的markdown內容：是filter出來的 List(String)
        #2. 表格的座標位置+行列數：pymu_table[bbox], pymu_table['rows'], pymu_table['cols'] List(Dict)
        #thought:
        #1. OCR解(depr.)
        #2. 對應表格的行列數
        #3. 若行列數相同，看位置(從上到下從左到右)
        #4. GPT護城河(先問是否相同，再進行解析)
        path_to_markdown = dict()
        markdown_to_rc = dict()
        #找出每一個markdown table有幾行幾列
        for table in table_markdown:
            num_rows, num_cols = check_row_column(table)
            markdown_to_rc[table] = (num_rows, num_cols)
        #先找出pymupdf的表格認為有幾行幾列，然後和markdown的表格比對
        for content in table_to_path:
            content = ast.literal_eval(content)
            content_rc = (content["rows"], content["columns"])
            content_rc2 = (content["rows"]+1, content["columns"])
            same_list = []
            for table_markdown in markdown_to_rc:
                if content_rc == markdown_to_rc[table_markdown] or content_rc2 == markdown_to_rc[table_markdown] :
                    same_list.append(table_markdown)
            if len(same_list) == 0:
                #就保持原樣，用markdown當作一般內容
                pass
            if len(same_list) == 1:
                path_to_markdown[table_to_path[str(content)]] = same_list[0]
                del markdown_to_rc[same_list[0]]
            if len(same_list) > 1:
                for table_markdown in same_list:
                    GPT_COUNT += 1
                    is_same = check_same_table(table_to_path[str(content)], table_markdown)
                    if is_same:
                        path_to_markdown[table_to_path[str(content)]] = table_markdown
                        del markdown_to_rc[table_markdown]
                        break
        return path_to_markdown, GPT_COUNT
                

    def illustrate_table(table_path, table_markdown_text, ocr_table=False): 
        print(f"illustrating table {table_path}, content: {table_markdown_text[0:10]}...")
        agent = Agent(role=table_role_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=False)
        local_prompt = table_local_prompt + table_markdown_text
        content = agent.run([local_prompt, {'image': output_dir+table_path}])
        if not ocr_table:
            markdown_to_html[table_markdown_text] = content
        else:
            ocr_table_list.append(content)

    def illustrate_image(image_path):
        print(f"illustrating image {image_path}")
        agent = Agent(role=image_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=True)
        content = agent.run([image_prompt, {'image':output_dir+image_path}])
        image_dict[content] = output_dir + image_path
        final_image_list.append(content)

    def illustrate_sole_table(sole_table): 
        print(f"illustrating table {sole_table}")
        agent = Agent(role=table_role_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=False)
        local_prompt = soletable_local_prompt
        content = agent.run([local_prompt, {'image': output_dir+table_path}])
        final_sole_table_list.append(content)



    def find_unique_table(text, pattern):
        markdown_content = []
        for match in pattern.finditer(text):
            markdown_content.append((match.start(), match.end(), match.group()))
        markdown_content.sort(key=lambda x: x[0])
        # 用於存儲唯一的表格信息
        matches = []
        for i, content in enumerate(markdown_content):
            is_unique = True
            for j, other_content in enumerate(markdown_content):
                if i != j and content[0] >= other_content[0] and content[1] < other_content[1]:
                    # 這個表格是另一個表格的子集，所以它不是唯一的
                    is_unique = False
                    break
            if is_unique:
                matches.append(content)
        return  matches
    GPT_COUNT = 0


    image_prompt = """你是一個具有豐富人類智慧的專業幫手機器人，你會用人類的角度來詮釋你所得到的圖片，請你使用繁體中文對這張圖片進行摘要"""

    table_role_prompt= """
    你現在是一位專注於製作HTML表格的工程師，你的任務是要畫出一個可以顯示的表格，並讓人類容易閱讀。
    """

    table_local_prompt = """
    你現在是一個製作HTML表格的工程師，你的任務是將圖片中表格的架構以及markdown的內容整合起來，並輸出一個html檔案，你必須要做到:
    1.你的輸出務必是完整的HTML格式，請不要省略輸出。
    2.請參考Markdown的內容，你只能對你得到的markdown表格內容和表格的圖片進行處理。
    3.如果表格和圖片內容略有不同時，請以表格圖片為主，因為markdown可能會有錯誤. markdown只是表格的初步表達，所以會有不足的地方需要你幫助改進．
    4.你可以適度地對表格內容進行修改，例如置中，增加邊框等等，但如果你選擇要做修改，請參考給予的圖片內容以及該圖片的標題，不得有任何不合理的地方。
    5.如果給予的圖片有跨欄置中，請你的html輸出也跨欄置中．
    
    6.請不要輸出除了html格式以外的任何內容，例如「以下是輸出的檔案內容」等等
    7.若有需要可以使用合併或分割儲存格。        
    8.標題非常重要，請你一定要照者提供的圖片中的標題格式，因為markdown的標題常常會出現錯誤，你在標題欄位可以不參考markdown的內容。
    9.標題可能有多行，你不需要將標題省略成一行，務必保持標題的完整性．請進行適度的修改

    最後再強調一次，請確保表格的完整性與是否與圖片中的格式相符，不能有跑版的狀況出現。
    你的工作對我來說很重要，請給我最好的表現.
    """

    soletable_local_prompt = """
    你現在是一個製作HTML表格的工程師，你的任務是將圖片中表格轉換成一個html檔案，你必須要做到:
    0.請一字不漏的看清楚圖片中的表格內容，並將其轉換成html格式。
    1.若有需要可以使用合併或分割儲存格。        
    2.你的輸出務必是完整的HTML格式，請不要省略輸出。
    3.你只能夠對你得到的圖片進行輸出，不要多加猜測。
    4.請注意會有無線表格的狀況，務必使得其結構更準確。
    5.請不要輸出除了html格式以外的任何內容，例如「以下是輸出的檔案內容」等等
    6. 標題非常重要，請你一定要照者圖片中的標題格式。
    最後再強調一次，請確保表格的完整性與是否與圖片中的格式相符，絕對不能有跑版的狀況出現。
    你的工作對我來說很重要，請給我最好的表現． I am watching you.
    """
    
        
            
    check_same_table_prompt = """你是一個具有豐富人類智慧的專業幫手機器人，你的工作是辨別這張圖片是否和這個表格相同，請你使用繁體中文回答"""
    table_checksame_prompt = """你將得到一個以markdown格式呈現的表格以及一張圖片，你的任務是要判斷這張圖片是否和這個表格相同
    若你判斷兩者相同，請回傳True
    若你判斷兩者不同，請回傳False
    請不要回傳其他內容，只回傳True或False
    \n\n文字內容:""" 

    # 用來存放每一頁的文字內容
    txt_pages = list()

    # 敘述內容：圖片位置
    image_dict = dict()
        
    # 創建parse_txt資料夾
    if not os.path.isdir("parse_txt"):
        os.makedirs("parse_txt")
    file_name = os.path.basename(pdf_path)
    final_output_path = "parse_txt/"+str(file_name.rstrip(".pdf"))+"_parse.txt"
    with open(final_output_path, "w") as file:
        pass

    if is_ocr:
        ocr = PaddleOCR(use_angle_cls=True, lang=ocr_lang)  #chinese_cht

    #把parse.py輸入的圖片和表格進行對應
    rect_loc_list = []
    for i in rects:
        for j in i:
            rect_loc_list.append(j)
    imagepath_to_loc = list(zip(image_paths, rect_loc_list)) 


    pdf_document = fitz.open(pdf_path)
    # 保存页面为图片
    markdown_to_html = dict()

    #實際上，我們傳進去的圖片內容都是從parse.py裡面得到的圖片，而pymupdf只是輔助
    doc_length = len(pymupdf4llm_list)
    for page_index,page_content in enumerate(pymupdf4llm_list):
        print(f"processing page {page_index+1}, total {doc_length} pages")
        ocr_table_list = []
        
        final_image_list = []
        final_sole_table_list = []
        
        # 尋找Pymupdf輸出的文字中的markdown部分
        pymupdf_original_text = page_content['text']
        table_markdown = filter_table(pymupdf_original_text)

        pymupdf_table_list = page_content['tables']
        pymupdf_image_list = page_content['images']
        filtered_imagepath_to_pageloc = list(filter(lambda x: x[0].startswith(str(page_index)+'_'), imagepath_to_loc))
        #parse_path_lict: 圖片位置對
        parse_path_list = [filename for filename in image_paths if filename.startswith(str(page_index)+'_')]
        if len(pymupdf_table_list) > 0:
            table_to_path = check_table_or_image(pymupdf_table_list, filtered_imagepath_to_pageloc, output_dir)
            print(table_to_path)
            path_to_markdown, GPT_COUNT = make_path_to_markdown(table_to_path, table_markdown, GPT_COUNT)
            # table_to_path : table pymupdf to 圖片path
            # path_to_markdown : 圖片path to文字

            table_path = list(table_to_path.values())

            for image_path in parse_path_list:
                if image_path not in table_path:
                    GPT_COUNT += 1
                    illustrate_image(image_path)

            for path in path_to_markdown:
                GPT_COUNT += 1
                illustrate_table(path, path_to_markdown[path])
            for path in final_sole_table_list:
                GPT_COUNT += 1
                illustrate_sole_table(path)

        # pymupdf抓到圖片但沒有抓到表格
        elif len(pymupdf_table_list) == 0 and len(parse_path_list)>0:
            for image_path in parse_path_list:
                GPT_COUNT += 1
                illustrate_image(image_path)
            
            
        else: #代表沒有找到表格但是說不定會有擷取到表格圖片，用OCR解
            # 如果pymupdf4llm沒有找到任何東西的話，就把資訊丟到最下面當補充，ＯＣＲ不能夠確定東西在哪裡，檔案另外存
            if not is_ocr:
                pass
            else:
                res = find_pic_images(output_dir, page_number=page_index)
                if res != []:
                    for image_ in res:
                        ocr_result = ocr.ocr(image_, cls=True)
                        first_chunk = ocr_result[0][0][1][0] 
                        last_chunk = ocr_result[0][-1][1][0]
                        fisrt_find = pymupdf_original_text.find(first_chunk[:20])
                        last_find = pymupdf_original_text.find(last_chunk[len(last_chunk)-20:])
                        # 表格內容
                        if last_find != -1 and fisrt_find != -1: 
                            ocr_text = pymupdf_original_text[fisrt_find:last_find+20]
                            GPT_COUNT += 1
                            illustrate_table(image_, ocr_text, ocr_table=True)
                        else:
                            #當作圖片處理
                            GPT_COUNT += 1
                            illustrate_image(image_)

        # 如果有OCR的表格，放在pymupdf_original_text的最下面
        if len(ocr_table_list) > 0:
            for ocrtable_index, ocr_table in enumerate(ocr_table_list):
                pymupdf_original_text += f"ocr table {ocrtable_index}:"
                pymupdf_original_text += f"{ocr_table}"
                pymupdf_original_text += f"end of ocr table {ocrtable_index}:"

        if len(final_sole_table_list) > 0:
            for soletable_index, sole_table in enumerate(final_sole_table_list):
                pymupdf_original_text += f"ocr table {soletable_index}:"
                pymupdf_original_text += f"{sole_table}"
                pymupdf_original_text += f"end of ocr table {soletable_index}:"
        
        # 如果有圖片，也是放在pymupdf_original_text的最下面
        if len(final_image_list) > 0:
            for index_image, image_content in enumerate(final_image_list):
                pymupdf_original_text += f"image {index_image}:"
                pymupdf_original_text += f"{image_content}"  #將HTML存進去txt
                pymupdf_original_text += f"end of image {index_image}"
        

        pattern = re.compile(
            r'(\|(?:[^\n]*\|)+\n'   # 匹配表格头部行
            r'\|(?:\s*[-:]+\s*\|)+\s*\n'  # 匹配表格分隔行
            r'(?:\|(?:[^\n]*\|)\n)+)',  # 匹配表格内容行
            re.DOTALL | re.MULTILINE
            )
        matches = find_unique_table(pymupdf_original_text, pattern)
        start_checkpoint = []
        end_checkpoint = []
        for i in matches:
            start_checkpoint.append(i[0])
            end_checkpoint.append(i[1])
        not_markdown_ranges = []
        prev_end = 0

        # 遍歷所有 checkpoint，找出範圍之外的區間
        for start, end in zip(start_checkpoint, end_checkpoint):
            if prev_end < start - 1:
                not_markdown_ranges.append((prev_end, start - 1))
            prev_end = end + 1
        not_markdown_ranges.append((prev_end, len(pymupdf_original_text)))

        for ranges in not_markdown_ranges:
            new_text = clean_string(pymupdf_original_text[ranges[0]:ranges[1]])
            pymupdf_original_text = pymupdf_original_text.replace(pymupdf_original_text[ranges[0]:ranges[1]], new_text)

        output_dict = {
            "page": page_index+1,
            "text": pymupdf_original_text
        }
        txt_pages.append(output_dict)
        print("--------------------")
    print("end of preprocess")
    return txt_pages, image_dict, markdown_to_html, GPT_COUNT
