import json
from json.decoder import JSONDecodeError
from examples import ast_example


"""将一个AST转换回source code"""

def get_json():
    """运行parser将一个js文件转换为ast并保存成.json格式，然后读入"""
    file = open('js_parser/test.json', 'r')
    string_ast = file.readlines()[0]
    ast = json.loads(string_ast)
    return ast

def get_string(ast):
    """给定一个AST，使用递归方式将其转换成一个string file"""

    def ast2code(token):
        #print(token)
        if type(token) == int:
            return

        type_info = token['type']
        return_string = ''

        if 'children' in token:
            child_list = token['children']
            if type_info == 'Program':
                for child in child_list:
                    return_string += ast2code(ast[child])
                    return_string += '\n'

            elif type_info == 'ForStatement':
                return_string = 'for('
                for index, child in enumerate(child_list):
                    if index == len(child_list)-2:
                        return_string += ast2code(ast[child])
                        return_string += ') '
                    elif index != len(child_list)-1:
                        return_string += ast2code(ast[child])
                        return_string += '; '
                    else:
                        return_string += '{\n'
                        return_string += ast2code(ast[child])
                return_string += '\n}'

            elif type_info == 'VariableDeclaration':
                return_string += 'var '
                for child in child_list:
                    return_string += ast2code(ast[child])

            elif type_info == 'VariableDeclarator':
                return_string += token['value'] + ' = '
                for child in child_list:
                    return_string += ast2code(ast[child])

            elif type_info == 'BinaryExpression':
                return_string += ast2code(ast[child_list[0]])
                return_string += token['value']
                return_string += ast2code(ast[child_list[1]])

            elif type_info == 'UpdateExpression':
                return_string += ast2code(ast[child_list[0]])
                return_string += token['value']

            elif type_info == 'BlockStatement':
                return_string += '    '
                for child in child_list:
                    return_string += ast2code(ast[child])

            elif type_info == 'ExpressionStatement':
                for child in child_list:
                    return_string += ast2code(ast[child])

            elif type_info == 'CallExpression':
                return_string += ast2code(ast[child_list[0]])
                return_string += '('
                return_string += ast2code(ast[child_list[1]])
                return_string += ')'

            elif type_info == 'MemberExpression':
                return_string += ast2code(ast[child_list[0]])
                return_string += '.'
                return_string += ast2code(ast[child_list[1]])

            else:
                raise KeyError('There is no non-terminal token: {}'.format(token))

        else:
            return_string += terminal_type(token)

        return return_string

    token = ast[0]
    return ast2code(token)


def terminal_type(token):
    """给定一个terminal token，将其转换"""
    type_info = token['type']
    if type_info == 'Identifier':
        return token['value']
    elif type_info == 'Property':
        return token['value']
    elif type_info == 'LiteralString':
        return '\"' + token['value'] + '\"'
    elif type_info == 'LiteralNumber':
        return token['value']
    else:
        print('unknown terminal error')
        return 'error'




if __name__ == '__main__':
    # for i in ast_example:
    #     print(i)
    ast = get_json()
    # for i in ast:
    #     print(i)
    string = get_string(ast)
    print(string)
