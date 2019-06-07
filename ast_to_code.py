import json
from json.decoder import JSONDecodeError
from examples import ast_example


"""将一个AST转换回source code"""

def get_json():
    file = open('js_parser/test.json', 'r')
    string_ast = file.readlines()
    ast = json.loads(string_ast)
    return ast


def ast2code(token):
    if 'children' in token:
        for child in token['children']:
            ast2code(child)
    else:
        if token['type'] == int:
            pass


def nonterminal_type(token):
    type_info = token['type']
    if type_info == 'Program':
        return
    elif type_info == 'ExpressionStatement':
        return


def terminal_type(token):
    type_info = token['type']
    if type_info == 'Identifier':
        return token['value']
    elif type_info == 'Property':
        return token['value']
    elif type_info == 'LiteralString':
        return token['value']
    elif type_info == 'LiteralNumber':
        return token['value']



if __name__ == '__main__':
    for i in ast_example:
        print(i)
