from Korpora import Korpora
import os

def download_korpora_datasets():
    try:
        # 단일 데이터셋만 시도
        Korpora.fetch(
            'namuwikitext',  # 나무위키 텍스트만 다운로드
            force_download=True
        )
        print("나무위키 데이터 다운로드 완료")
        
    except Exception as e:
        print(f"다운로드 중 오류 발생: {str(e)}")
        raise e  # 오류 메시지를 볼 수 있도록 예외를 다시 발생시킴

if __name__ == "__main__":
    download_korpora_datasets() 