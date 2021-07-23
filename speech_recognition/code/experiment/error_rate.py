import numpy as np

# edit distance
def levenshtein_distance(ref, hyp):
    
    m = len(ref) # reference
    n = len(hyp) # hypothesis

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use 0 (min(m,n)) space
    distance = np.zeros((2, n+1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0, n+1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m+1):

        prev_row_idx = (i-1)%2
        cur_row_idx = i%2

        distance[cur_row_idx][0] = i

        for j in range(1,n+1):
            if ref[i-1] == hyp[j-1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j-1]
            else:
                s_num = distance[prev_row_idx][j-1] + 1
                i_num = distance[cur_row_idx][j-1] + 1
                d_num = distance[prev_row_idx][j] + 1

                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m%2][n]

def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    if ignore_case == True:
        reference = reference.upper()
        hypothesis = hypothesis.upper()
    
    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = levenshtein_distance(ref_words, hyp_words)

    return float(edit_distance), len(ref_words)

def char_errors(reference, hypothesis, ignore_case=False, delimiter=' ', remove_space=False):
    if ignore_case == True:
        reference = reference.upper()
        hypothesis = hypothesis.upper()

    join_char=' '

    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(delimiter)))
    hypothesis = join_char.join(filter(None, hypothesis.split(delimiter)))

    edit_distance = levenshtein_distance(reference, hypothesis)

    return float(edit_distance), len(reference)

# word error rate
'''
WER = (Sw + Dw + Iw) / Nw

Sw는 대체 된 단어의 수
Dw는 삭제 된 단어의 수
Iw는 삽입 된 단어의 수
Nw는 참조의 단어 수
'''

def WER(reference, hypothesis, ignore_case=False, delimiter=' '):

    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case, delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance)/ref_len

    return wer

# character error rate
'''
CER = (Sc + Dc + Ic) / Nc

Sc는 대체 된 문자의 수
Dc는 삭제 된 문자의 수
Ic는 삽입 된 문자의 수
Nc는 참조의 문자 수
'''

def CER(reference, hypothesis, ignore_case=False, delimiter=' ', remove_space=False):

    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case, delimiter, remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance)/ref_len

    return cer    