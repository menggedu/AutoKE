import enum
import re
# from graph_gen import *

from dataclasses import dataclass

class TokenType(enum.Enum):
    T_NUM = 0
    T_PLUS = 1
    T_MINUS = 2
    T_MULT = 3
    T_DIV = 4
    T_LPAR = 5
    T_RPAR = 6
    T_VARIABLE = 7
    T_DIFF = 8
    T_FUNC = 9
    T_POW = 10
    T_END = 11

# @dataclass
# class Token:
# 	type: TokenType
# 	value: any = None

# 	def __repr__(self):
# 		return self.type.name + (f":{self.value}" if self.value != None else "")
class Node:
    def __init__(self, token_type, value=None):
        self.token_type = token_type
        self.value = value
        self.left = None
        self.right = None
        self.children = []
        
        self.result = None #result cache for operator
        
    def __repr__(self) -> str:
        return self.token_type.name+f':{self.value}'
    
mappings = {
    '+': TokenType.T_PLUS,
    '-': TokenType.T_MINUS,
    '*': TokenType.T_MULT,
    '/': TokenType.T_DIV,
    '(': TokenType.T_LPAR,
    ')': TokenType.T_RPAR,
    'diff': TokenType.T_DIFF,
    '^':TokenType.T_POW
}
variables = ['u','x','t','e','y','a','b']

funcs = {
    "F1": 'f * K * diff(u,x)',
    'F2':'u * diff(u,x)',
    "F3": 'f * K * diff(u,y)', 
    
}
sub_funcs = {'f': ['f','u'],
         'K': ['K', 'x']}
for v in variables:
    mappings[v] = TokenType.T_VARIABLE
for v in funcs.keys():
    mappings[funcs[v]] = TokenType.T_FUNC
for v in sub_funcs.keys():
    mappings[sub_funcs[v][0]] = TokenType.T_FUNC    
if __name__=='__main__':
    print([Node(TokenType.T_NUM,10),Node(TokenType.T_VARIABLE,'A')])