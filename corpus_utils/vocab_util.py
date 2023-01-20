class Vocab:
    def __init__(self):
        self.PAD = 0
        self.SOS = 1
        self.EOS = 2
        self.tok2indx = {}
        self.indx2tok = {self.PAD:"PAD", self.SOS:"SOS", self.EOS:"EOS"}
        self.tokens_size = 3
        # self.pieces records function names in order to avoid further separation
        self.pieces = set()


    def build_vocab(self, tokens):
        for tok in tokens:
            if tok not in self.tok2indx:
                self.tok2indx[tok] = self.tokens_size
                self.indx2tok[self.tokens_size] = tok
                self.tokens_size += 1
                if len(tok) > 1:
                    self.pieces.add(tok)
        self.pieces = sorted(self.pieces, key=lambda x:len(x), reverse=True)


    def transform_to_index(self, string):
        #toks = []
        ids = []
        pointer = 0
        string = string.lower()
        while pointer < len(string):
            for p in self.pieces:
                if string[pointer:].startswith(p):
                    #toks.append(p)
                    ids.append(self.tok2indx[p])
                    pointer += len(p)
                    continue
            #toks.append(string[pointer])
            ids.append(self.tok2indx[string[pointer]])
            pointer += 1
        return ids
                    
            
    
