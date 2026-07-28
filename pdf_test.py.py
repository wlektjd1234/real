# 1. 아까 다운받은 'pymupdf4llm' 공구 상자를 열어서 꺼냅니다.
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter

print("PDF 읽는 중... 잠시만 기다려주세요!")

# 2. 내 PDF 파일 이름을 정확히 적어줍니다.
file_name = "R2_KR514X450008.pdf"

# 3. 공구 상자 안의 'to_markdown'이라는 마법 지팡이를 써서 텍스트를 뽑아냅니다.
md_text = pymupdf4llm.to_markdown(file_name)

print("\n이제 텍스트를 목차별로 자릅니다...")

# 4. 가위에게 "어디를 기준으로 자를지" 알려줍니다. (샵(#) 기호 기준)
headers_to_split_on = [
    ("#", "대분류"),
    ("##", "중분류"),
    ("###", "소분류"),
]

# 5. 가위 세팅 완료!
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# 6. 전체 글(md_text)을 가위로 싹둑싹둑 자릅니다.
chunks = markdown_splitter.split_text(md_text)

# 7. 총 몇 덩어리로 잘렸는지, 첫 번째 덩어리 내용은 뭔지 확인해 봅니다.
print(f"\n총 {len(chunks)}개의 덩어리로 예쁘게 잘렸습니다!")
print("--- 첫 번째 덩어리 샘플 ---")
print("꼬리표(목차 위치):", chunks[0].metadata)
print("내용 일부:", chunks[0].page_content[:150])