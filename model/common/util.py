import sys
sys.path.append('../..')
from model.common.np import *


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_idx = {}
    idx_to_word = {}

    for word in words:
        if word not in word_to_idx:
            new_id = len(word_to_idx)
            idx_to_word[new_id] = word
            word_to_idx[word] = new_id

    corpus = np.array([word_to_idx[w] for w in words])

    return corpus, word_to_idx, idx_to_word


def cos_similarity(x, y, eps=12-8):
    """
    코사인 유사도는 두 벡터간의 코사인 각도를 이용하여 구할 수 있는 두 벡터의 유사도.
    두 방향이 완전 동일 -> 1
    90각도 차이 -> 0
    180도 반대방향 -> -1
    :param x:
    :param y:
    :param eps:
    :return:
    """
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def most_similar(q, word_to_idx, idx_to_word, word_matrix, top=5):
    '''
    유사단어 검색
    1. 워드 벡터로 변경
    forloop로 본인 제외 유사도 계싼
    :param q:
    :param word_to_idx:
    :param idx_to_word:
    :param word_matrix:
    :param top:
    :return:
    '''
    if q not in word_to_idx:
        return
    q_id = word_to_idx[q]
    q_vec = word_matrix(q_id)

    # 코사인 유사도 계산
    vocab_size = len(idx_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(q_vec, word_matrix[i])

    print(similarity)

    count = 0
    for i in (-1 * similarity).argsort():
        if idx_to_word[i] == q:
            continue
        print(' %s: %s' % (idx_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def clip_grads(grads, max_norm):
    """

    :param grads:
    :param max_norm:
    :return:
    """
    total_norm = 0
    for grad in grads:  # 전체 기울기에 대해 norm 을 구함.
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def convert_one_hot(corpus, vocab_size):
    '''
    one-hot으로 표현
    :param corpus: 1D혹은 (batch로 연결된 2차원 넘파이 배열)
    :param vocab_size: 어휘 수
    :return:원핫 표현 (2차 혹은 3차원 넘파이 배열)
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:  # 문장 1개
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_in in enumerate(corpus):
            one_hot[idx, word_in] = 1

    elif corpus.ndim == 2:  # 문장이 C개 들어옴
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):  # corpus 들
            for idx_1, word_id in enumerate(word_ids):  # word 들
                one_hot[idx_0, idx_1, word_id] = 1  # corpusId, wordId, 단어 id

    else:
        raise ValueError('embedding layer dimension issue')

    return one_hot


def create_contexts_target(corpus, window_size=1):
    """
    맥락과 타깃 생성
    :param corpus: 단어 id 목록
    :param window_size: word2vec 윈도우 크기
    :return:
    """
    target = corpus[window_size:-window_size]  # 앞뒤로 참조할 단어가 있는 idx들만
    contexts = []
    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1): # window_size까지 보려면 +1
            if t == 0:  # 자기 자신은 제외
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)  # words별 참조 임베딩 리스트 저장
    return np.array(contexts), np.array(target)
