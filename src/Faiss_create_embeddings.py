import os
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# 엑셀 파일 목록 및 시트 목록
excel_files = [
    ("data_source/세무사데이터.xlsx", ["Sheet1"]),
]

# 모든 데이터를 하나의 리스트에 저장
documents = []

# 파일별로 모든 시트를 읽어 Document로 변환
for file, sheets in excel_files:
    for sheet in sheets:
        # 시트 데이터 읽기
        df = pd.read_excel(file, engine='openpyxl', sheet_name=sheet)
        
        # 각 행을 Document로 변환하여 리스트에 추가
        for index, row in df.iterrows():
            # content는 제목과 본문_원본을 포함
            content = f"제목: {row['제목']}\n본문: {row['본문_원본']}"
            
            # metadata는 파일명과 문서명을 포함
            metadata = {
                "파일명": row["파일명"],
                "문서명": row["문서명"],
                "source": f"{file} - {sheet}"
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
print(f"documents: {documents}")

# # 임베딩 생성
# embedding_model = OpenAIEmbeddings()
# vectorstore = FAISS.from_documents(documents=documents, embedding=embedding_model)

# # 벡터 저장소와 메타데이터를 파일로 저장
# vectorstore.save_local("faiss_index")
# print("모든 시트의 임베딩이 성공적으로 저장되었습니다.")
