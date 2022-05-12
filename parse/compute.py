import sys
import torch
from parse.tokens import *
from parse.lexer import Lexer
from parse.parser_ import Parser
import pdb
from parse.utils import *
from parse.graph import *

def diff(x,u,order=1):
    # pdb.set_trace()
    grad = torch.autograd.grad(outputs=u,inputs = x,
                               grad_outputs = torch.ones_like(u),
                               retain_graph = True, create_graph=True)[0]
    for _ in range(order-1):
        grad = torch.autograd.grad(outputs=grad,inputs = x,
                               grad_outputs = torch.ones_like(u),
                               retain_graph = True, create_graph=True)[0]
    return grad

def power(a,b):
    if a.value == 'e':
        return torch.exp(b)
    return torch.pow(a,b)

operations = {
    TokenType.T_PLUS: torch.add,
    TokenType.T_MINUS: torch.sub,
    TokenType.T_MULT: torch.mul,
    TokenType.T_DIV: torch.div,
    TokenType.T_DIFF: diff,
    TokenType.T_POW: power
}

def compute(node):
    if node.token_type in [TokenType.T_NUM, TokenType.T_VARIABLE] :
        if node.value in ['e']:#['x','t','e','y']:
            return node.value
        # pdb.set_trace()
        return eval(node.value)
    # pdb.set_trace()
    left_result = compute(node.children[0])
    right_result = compute(node.children[1])
    operation = operations[node.token_type]
    
    result = operation(left_result, right_result)
    return result

class Interpreter:
    def __init__(self, ascii_rep):
        self.ascii_rep = ascii_rep
        lexer = Lexer(ascii_rep)
        tokens = lexer.generate_tokens()
        parser = Parser(tokens)
        self.ast = parser.parse()
        
        
    
    def postOrderTraverse(self, u,x,t):
        compute(self.ast)
        
    


if __name__ == '__main__':
    input_s = 'diff(F2,x) - diff(u,t)'
    input_s = '1 * u * diff(u,x) + 0.0025 * diff(u,x,3) - diff(u,t)'
    lex = Lexer(input_s)
    tokens = lex.generate_tokens()
    lex2 = Lexer(input_s)
    tokens2 = lex.generate_tokens()
    tokens_copy = list(tokens2)
    print(tokens_copy)
    parser = Parser(tokens)
    ast = parser.parse()
    x = torch.rand(10,4)
    t = torch.rand(10,4)
    x.requires_grad = True
    t.requires_grad = True
    u = (x**2)+t
    x2 = x.clone()
    t2 =t.clone()
    
    u2 = (x2**2)+t2
    # pdb.set_trace()
    result = compute(ast)

    # result2 = diff(x2, u2*diff(x2,u2))-diff(t2,u2) 
    result2 = u2*diff(x2,u2)+0.0025*diff(x2,u2,3)-diff(t2,u2)
    print(result-result2)