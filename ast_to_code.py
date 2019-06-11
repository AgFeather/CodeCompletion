import json
from json.decoder import JSONDecodeError
from examples import ast_example


"""convert an AST back to source code"""

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

            elif type_info == 'FunctionExpression':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'ObjectExpression':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'UnaryExpression':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'DoWhileStatement':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'UnaryExpression':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'AssignmentPattern':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'AssignmentExpression':
                assert len(child_list) == 2
                return_string += ast2code(ast[child_list[0]]) + ' = '
                return_string += ast2code(ast[child_list[1]])

            elif type_info == 'SequenceExpression':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'LogicalExpression':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'UnaryExpression':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'ThrowStatement':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'LabeledStatement':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'CatchClause':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'TryStatement':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'NewExpression':
                child = child_list[0]
                return_string += 'new ' + ast2code(ast[child])

            elif type_info == 'ForInStatement':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'ArrayExpression':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'Property':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'WhileStatement':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'ArrayAccess':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'FunctionDeclaration':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'ReturnStatement':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'ConditionalExpression':
                raise KeyError('There is no non-terminal token: {}'.format(token))

            elif type_info == 'SwitchStatement':
                for i, child in enumerate(child_list):
                    if i == 0:
                        return_string += 'switch (' + ast2code(ast[child]) + ')\n{'
                    else:
                        return_string += ast2code(ast[child])
                return_string += '\n}'

            elif type_info == 'SwitchCase':
                assert len(child_list) == 2
                return_string += 'case ' + ast2code(ast[child_list[0]]) + ':'
                return_string += ast2code(ast[child_list[1]])

            elif type_info == 'IfStatement':
                raise KeyError('There is no non-terminal token: {}'.format(token))







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
                assert len(child_list) == 2
                return_string += ast2code(ast[child_list[0]])
                return_string += token['value']
                return_string += ast2code(ast[child_list[1]])
            elif type_info == 'UpdateExpression':
                assert len(child_list) == 1
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
                for i, child in enumerate(child_list):
                    if i == 0:
                        return_string += ast2code(ast[child])
                        return_string += '('
                    else:
                        return_string += ast2code(ast[child])
                return_string += ')'
            elif type_info == 'MemberExpression':
                assert len(child_list) == 2
                return_string += ast2code(ast[child_list[0]])
                return_string += '.'
                return_string += ast2code(ast[child_list[1]])
            else:
                raise KeyError('There is no non-terminal token: {}'.format(token))

        elif type_info == 'BreakStatement':
            return_string += '\nbreak\n'

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
        print('error', token)
        raise KeyError('terminal error')
        return 'error'




if __name__ == '__main__':
    # for i in ast_example:
    #     print(i)
    ast = get_json()
    for i in ast: print(i)
    string = get_string(ast)
    print(string)
