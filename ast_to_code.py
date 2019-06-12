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

hole_index = -1

def get_string(ast):
    """给定一个AST，使用递归方式将其转换成一个string file"""

    def ast2code(token):
        if type(token) == int:
            return
        if token['id'] == hole_index:
            print('The location of the hole')

        type_info = token['type']
        return_string = ''

        if 'children' in token:
            child_list = token['children']
            if type_info == 'Program':
                for child in child_list:
                    return_string += ast2code(ast[child])
                    return_string += '\n'

            elif type_info == 'ObjectExpression':
                return_string += '{'
                for child in child_list:
                    return_string += ast2code(ast[child]) + ','
                return_string += '}'

            elif type_info == 'UnaryExpression':
                assert len(child_list) == 1
                return_string += token['value'] + ' '
                return_string += ast2code(ast[child_list[0]])

            elif type_info == 'DoWhileStatement':
                assert len(child_list) == 2
                return_string += 'do' + ast2code(ast[child_list[1]])
                return_string += 'while (' + ast2code(ast[child_list[0]]) + ')'


            elif type_info == 'FunctionExpression':
                raise KeyError('There is no non-terminal token: {}'.format(token))
            elif type_info == 'AssignmentPattern':
                raise KeyError('There is no non-terminal token: {}'.format(token))
            elif type_info == 'SequenceExpression':
                raise KeyError('There is no non-terminal token: {}'.format(token))
            elif type_info == 'ThrowStatement':
                assert len(child_list) == 1
                return_string += 'throw ' + ast2code(ast[child_list[0]])
            elif type_info == 'CatchClause':
                assert len(child_list) == 2
                return_string += 'catch(' + ast2code(ast[child_list[0]]) + ')'
                return_string += ast2code(ast[child_list[1]])
            elif type_info == 'TryStatement':
                assert len(child_list) == 3
                return_string += 'try '
                for child in child_list:
                    return_string += ast2code(ast[child])



            elif type_info == 'AssignmentExpression':
                assert len(child_list) == 2
                return_string += ast2code(ast[child_list[0]]) + ' = '
                return_string += ast2code(ast[child_list[1]])

            elif type_info == 'LogicalExpression':
                assert len(child_list) == 2
                return_string += ast2code(ast[child_list[0]])
                return_string += ' ' + token['value'] + ' '
                return_string += ast2code(ast[child_list[1]])

            elif type_info == 'LabeledStatement':
                assert len(child_list) == 1
                return_string += token['value'] + ':' + ast2code(ast[child_list[0]])

            elif type_info == 'NewExpression':
                for i, child in enumerate(child_list):
                    if i == 0:
                        return_string += 'new ' + ast2code(ast[child]) + '('
                    elif i == 1:
                        return_string += '(' + ast2code(ast[child]) + ', '
                    else:
                        return_string += ast2code(ast[child]) + ', '
                #if len(child_list) > 1: # 说明该new的对象存在初始化参数
                return_string +=  ')'


            elif type_info == 'ForInStatement':
                assert len(child_list) == 3
                return_string += 'for (' + ast2code(ast[child_list[0]]) + ' in '
                return_string += ast2code(ast[child_list[1]]) + ')'
                return_string += ast2code(ast[child_list[2]])

            elif type_info == 'ArrayExpression':
                return_string += '['
                for child in child_list:
                    return_string += ast2code(ast[child]) + ', '
                return_string += ']'

            elif type_info == 'Property':
                assert len(child_list) == 1
                return_string += token['value'] + ':'
                return_string += ast2code(ast[child_list[0]])

            elif type_info == 'WhileStatement':
                assert len(child_list) == 2
                return_string += 'while (' + ast2code(ast[child_list[0]]) + ')'
                return_string += ast2code(ast[child_list[1]])

            elif type_info == 'ArrayAccess':
                assert len(child_list) == 2
                return_string += ast2code(ast[child_list[0]]) + '['
                return_string += ast2code(ast[child_list[1]]) + ']'

            elif type_info == 'FunctionDeclaration':
                return_string += 'function '
                for i, child in enumerate(child_list):
                    if i == 0:
                        return_string += ast2code(ast[child]) + '('  # function name
                    elif i != len(child_list)-1:
                        return_string += ast2code(ast[child]) + ', '
                    else:
                        return_string += ') ' + ast2code(ast[child])

            elif type_info == 'ReturnStatement':
                assert len(child_list) == 1
                return_string += 'return '
                return_string += ast2code(ast[child_list[0]])

            elif type_info == 'ConditionalExpression':
                assert len(child_list) == 3
                return_string += ast2code(ast[child_list[0]]) + ' ? '
                return_string += ast2code(ast[child_list[1]]) + ' : '
                return_string += ast2code(ast[child_list[2]])

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
                return_string += 'if ('
                return_string += ast2code(ast[child_list[0]]) + ') '
                return_string += ast2code(ast[child_list[1]])
                if len(child_list) == 2:
                    pass
                elif len(child_list) == 3:
                    return_string += 'else ' + ast2code(ast[child_list[2]])

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
                        return_string += ast2code(ast[child])

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
                return_string += ' ' + token['value'] + ' '
                return_string += ast2code(ast[child_list[1]])
            elif type_info == 'UpdateExpression':
                assert len(child_list) == 1
                return_string += ast2code(ast[child_list[0]])
                return_string += token['value']
            elif type_info == 'BlockStatement':
                return_string += '{\n'
                for child in child_list:
                    return_string += ast2code(ast[child]) + '\n'
                return_string += '}'
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
            return_string += 'break'
            if 'value' in token:
                return_string += ' ' + token['value']
        elif type_info == 'EmptyStatement':
            return_string +=''
        elif type_info == 'ContinueStatement':
            return_string += 'continue'
        elif type_info == 'VariableDeclarator':
            return_string += token['value'] + ', '
        elif type_info == 'BlockStatement':
            return_string += "{}"
        else:
            return_string += terminal_type(token)

        return return_string

    print('\n')
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
    elif type_info == 'LiteralNull':
        return token['value']
    elif type_info == 'LiteralBoolean':
        return token['value']
    else:
        print('error', token)
        raise KeyError('terminal error', str(token))




if __name__ == '__main__':
    # for i in ast_example:
    #     print(i)
    ast = get_json()
    for i in ast: print(i)
    string = get_string(ast)
    print(string)
