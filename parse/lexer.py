import enum
import re
# import sys
# sys.path.append("../")
# from parse.graph import *
from parse.tokens import *
from parse.utils import *
WHITESPACE = ' \n\t'
DIGITS = '0123456789'


class Lexer:
    def __init__(self, text, mappings, func_config):
        self.mappings = mappings
        self.text = text.split()
        self.func_config = func_config
        self.funcs = func_config['funcs'] if 'funcs' in func_config else None
         
   
    def generate_tokens(self):
        
        for s in self.text:
            if s in self.mappings:
                token_type = self.mappings[s]
                yield Node(token_type, value=s)
            elif re.match(r'\d+(?:\.\d+)?',s):
                yield Node(TokenType.T_NUM, value = s)
            elif 'diff' in s:
                diff_tokens = self.generate_diff(s)
                for t in diff_tokens:
                    yield t
            else:
                raise Exception('Invalid token: {}'.format(s))
        yield Node(TokenType.T_END)
            
    def generate_diff(self, exp):
        params = exp[5:-1].split(',')# retrive params from diff()
        tokens = []
        new_param =[]

        if self.funcs is not None:
            param0 = params[0]
            for f in self.funcs.keys():
                if f in param0:
                    param0 = param0.replace(f,self.funcs[f])
            
            params[0] = param0
        compoun_diff = False


        if "*" in params[0]:
            compoun_diff = True
        
        #['x','diff','u']

        if len(params) == 2:
            tokens.append(Node(TokenType.T_VARIABLE, value=params[1]))
            tokens.append(Node(TokenType.T_DIFF, value='diff'))
            
            if "*" in params[0]:
                # diff 嵌套 exist
                lex = Lexer(params[0], self.mappings, self.func_config)
                tokens_ = list(lex.generate_tokens())
                tokens.append(tokens_[:])
                compoun_diff = True
                
            if not compoun_diff:
                tokens.append(Node(TokenType.T_VARIABLE, value=params[0]))
            # for t in tokens:
            #     yield t
            return tokens

        #['x', 'diff', 'x','diff','u']
        order = int(params[-1])
        for _ in range(order):

            tokens.insert(0,Node(TokenType.T_DIFF, value='diff'))
            tokens.insert(0,Node(TokenType.T_VARIABLE, value=params[1]))
            
        if "*" in params[0]:
                # diff 嵌套 exist
            lex = Lexer(params[0])
            tokens_ = list(lex.generate_tokens())
            tokens.append(tokens_[:])
            compoun_diff = True
        if not compoun_diff:
            tokens.append(Node(TokenType.T_VARIABLE, value=params[0]))
        # for t in tokens:
        #     yield t
        return tokens           

            
    

if __name__ == '__main__':
    input_s = ['diff(u,x,2)']
    # input_s = ['2 * u * diff(u,x,2)','diff(u * diff(u,x))', 'diff(f * k * diff(u,x),u)']
    input_s = ['2 * u * diff(u,x,2)','diff(F1,x)', 'diff(F2,x)','diff(u,x,2)',]
    for s in input_s:
        print(s)
        lex = Lexer(s)
        out = list(lex.generate_tokens())
        print(out)
    # import pdb;pdb.set_trace()