from gensim.models import Word2Vec  # word2vec
from konlpy.tag import Mecab        # 형태소 분석기
import kss                          # 문장 분리기
from stop_words import stop_words   # 불용어 리스트


class ReviewPreprocessor():
    
    def __init__(self, w2v_model_fname: str, keywords: list):
        self.stop_words = stop_words
        self.tokenize = Mecab().pos
        self.w2v_model = Word2Vec.load(w2v_model_fname)
        self.splitter = kss.split_sentences
        self.keywords = keywords

        self.similar_words = {}
        for keyword in self.keywords:
            temp = self.w2v_model.wv.most_similar(positive=[keyword], topn=15)
            self.similar_words[keyword] = dict(temp)

    def process(self, text: str) -> list:
        """
        text를 문장 단위로 분리, 그 후 라벨링 및 점수를 부여한다.
        """

        # text를 문장 단위로 분리
        sentences = self.splitter(text)

        # 분리한 문장들을 토큰화 및 불용어 제거
        text_refined = []
        for sentence in sentences:
            tokenized = self.tokenize(sentence)

            tokenized_refined = []
            for token in tokenized:
                if token[1] not in self.stop_words:
                    tokenized_refined.append(token[0])

            item = [tokenized_refined, sentence]
            text_refined.append(item)

        ## 문장 하나하나에 라벨링하는 작업
        keyword_score = 0.8
        for i, e in enumerate(text_refined):
            label = ""
            top_score = -1.0
            for keyword in self.keywords:
                keyword_flag = False
                score = 0.0
                for token in e[0]:
                    # token이 keyword이면 고정 점수(keyword_score) 부여
                    if token == keyword and keyword_flag == False :
                        score = score + keyword_score
                        keyword_flag = True
                    # token이 keyword가 가진 유사단어이면, 점수 추가
                    elif token in self.similar_words[keyword].keys():
                        score = score + self.similar_words[keyword][token]
                # 가장 높은 score로 label 갱신
                if score > top_score:
                    label = keyword
                    top_score = score
            # top_score가 keyword_score보다 작으면 '기타'로 라벨링
            if top_score < keyword_score:
                label = "기타"

            text_refined[i].append(label)
            text_refined[i].append(top_score)

        for x in text_refined:
            del x[0]
        
        return text_refined

        


if __name__ == "__main__":
    
    keywords = ["디자인", "화면", "무게", "발열", "사양", "배터리", "소음", "배송"]

    processor = ReviewPreprocessor("w2vmodel_notebook", keywords)
    
    print(processor.process("배송이 참 빠르네요. 근데 화면 크기가 좀... 배터리는 넉넉하네요.만족합니당"))
    print(processor.process("팬소리 심합니다. 엄청 뜨거움"))

        