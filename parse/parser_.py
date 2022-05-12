from parse.tokens import *
from parse.lexer import Lexer
from parse.utils import *
from parse.graph import *

class Parser:
    def __init__(self,tokens):
        self.tokens = iter(tokens)

        self.advance()
        
    def advance(self):
        try:
            self.current_token = next(self.tokens) #self.tokens.pop(0)
        except StopIteration:
            self.current_token = None
    def raise_error(self):
        return Exception("illegal syntax")
    
    def parse(self):
        result = self.parse_e1()
        match(self.current_token, TokenType.T_END)

        return result
    
    def parse_e1(self):
        left_node = self.parse_e2()
        # import pdb;pdb.set_trace()
        while self.current_token is not None and self.current_token.token_type in [TokenType.T_PLUS, TokenType.T_MINUS]:
            node = self.current_token
            self.advance()
            node.children.append(left_node)
            node.children.append(self.parse_e2())
            left_node = node 
        return left_node
    def parse_e2(self):
        left_node = self.parse_e3()
        # import pdb;pdb.set_trace()
        while self.current_token is not None and self.current_token.token_type in [TokenType.T_MULT, TokenType.T_DIV]:
            node =  self.current_token
            self.advance()
            node.children.append(left_node)
            node.children.append(self.parse_e3())
            left_node = node

        return left_node
    
    def parse_diff(self):

        # import pdb;pdb.set_trace()
        parser = Parser(self.current_token)
        left_node = parser.parse()
        return left_node

    def parse_e3(self):
        left_node= self.parse_e4()
        self.advance()
        while  self.current_token is not None and self.current_token.token_type == TokenType.T_DIFF:
            node = self.current_token        
            self.advance()
            node.children.append(left_node)
            node.children.append(self.parse_e3())
            left_node = node
        return left_node
    
    def parse_e4(self):
            # import pdb;pdb.set_trace()
        if isinstance(self.current_token, list):
            return self.parse_diff()
        if self.current_token.token_type in [TokenType.T_NUM, TokenType.T_VARIABLE,TokenType.T_FUNC]:
            return  self.current_token

        match( self.current_token, TokenType.T_LPAR) #flip out ()
        expression = self.parse_e1()
        self.advance()
        match( self.current_token, TokenType.T_RPAR)
        self.advance()
        return expression
    
if __name__ == '__main__':
    #single_pahse
    input_s = 'diff(F1,x) + diff(F3,y) - diff(u,t)'
    #vgc
    input_s = 'diff(F1,x)'
    #kdv
    input_s = 'a * u * diff(u,x) + b * diff(u,x,3) - diff(u,t)'
    input_s = '1 * u * diff(u,x) + 0.0025 * diff(u,x,3) - diff(u,t)'
    lex = Lexer(input_s)
    tokens = lex.generate_tokens()
    lex2 = Lexer(input_s)
    tokens2 = lex.generate_tokens()
  
    # tokens = [[T_VARIABLE:x, T_DIFF:diff, T_VARIABLE:x, T_DIFF:diff, T_VARIABLE:u], T_END:None]
    tokens_copy = list(tokens2)
    # print(tokens_copy)
    parser = Parser(tokens)
    ast = parser.parse()
    label(ast)
    to_graphviz(ast)
