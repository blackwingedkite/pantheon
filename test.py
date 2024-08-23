# import re

#  原始字串
# text = """
# 這是一些文字，包含了特殊格式的內容。
# ![](測試文件.pdf-1-0.png)
# 這是另一段文字，包含了另一個特殊格式的內容。
# ![](測試文件.pdf-2-0.png)
# """

# # 指定的替換內容列表
# replacement_list = ["替換內容4444", "替換內容111"]

# # 定義正則表達式來匹配特殊格式的內容
# pattern = r'!\[\]\((.*?)\)'

# # 找到所有匹配的內容
# matches = re.findall(pattern, text)

# # 確保替換列表的數量和匹配數量一致
# if len(matches) != len(replacement_list):
#     raise ValueError("替換內容列表的數量與匹配的數量不一致")

# # 替換匹配的內容為替換列表中的對應內容
# for i, match in enumerate(matches):
#     text = text.replace(f"![]({match})", replacement_list[i], 1)

# print(text)
name = "tat_docs/" + "tttssshello" + ".pdf"
pdf_path_idef = name.lstrip("tat_docs/").rstrip(".pdf")
print(pdf_path_idef)