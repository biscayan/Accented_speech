'''
The outputs of the network are the graphemes of each language. 
At each output time-step t, the RNN makes a prediction over characters.
In English we have `t ∈ {a, b, c, . . . , z,space, apostrophe, blank}
'''

###maps characters to integers and vice versa

class TextTransform:
    def __init__(self):
        char_map_str="""
        ' 0
        <SPACE> 1
        A 2
        B 3
        C 4
        D 5
        E 6
        F 7
        G 8
        H 9
        I 10
        J 11
        K 12
        L 13
        M 14
        N 15
        O 16
        P 17
        Q 18
        R 19
        S 20
        T 21
        U 22
        V 23
        W 24
        X 25
        Y 26
        Z 27
        """

        self.char_map = {}
        self.index_map = {}

        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        
        self.index_map[1] = ' '

    #Use a character map and convert text to an integer sequence
    def text_to_int(self, text):

        int_sequence = []

        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            
            int_sequence.append(ch)
        
        return int_sequence

    #Use a character map and convert integer labels to an text sequence
    def int_to_text(self, labels):

        string = []

        for i in labels:
            string.append(self.index_map[i])

        return ''.join(string).replace('<SPACE>',' ')