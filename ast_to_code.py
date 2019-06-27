import json
from json.decoder import JSONDecodeError
import pickle
import os

"""convert an AST back to source code"""

def get_json():
    """运行parser将一个js文件转换为ast并保存成.json格式，然后读入"""
    os.system('node js_parser/js_parser.js js_parser/helloworld.js')
    file = open('js_parser/test.json', 'r')
    string_ast = file.readlines()[0]
    ast = json.loads(string_ast)
    return ast

def get_test_ast_with_pickle(is_terminal):
    if is_terminal:
        files_path = 'temp_data/predict_compare/terminal/'
    else:
        files_path = 'temp_data/predict_compare/non_terminal/'
    file_list = os.listdir(files_path)
    for file in file_list:
        if file.startswith('.'):
            continue
        info_list = pickle.load(open(files_path + file, 'rb'))
        for one_ast in info_list:
            ast = one_ast['ast']
            expect_index = one_ast['expect_index']
            ori_pred = one_ast['ori_pred']
            embed_pred = one_ast['embed_pred']
            expect_token = one_ast['expectation']
            if not isinstance(expect_index, int): # 跳过对EMTPY的预测
                continue
            yield ast, expect_index, ori_pred, embed_pred, expect_token

def get_test_ast():
    """读取lstm的预测结果中的AST，并将AST转换成源码，在转换的同时标注出hole的位置"""
    file = open('temp_data/predict_compare/tt_compare.txt')
    ast = file.readline()
    while ast:
        hole_index = file.readline()
        ori_pred = file.readline()
        embed_pred = file.readline()
        ast = ast.split(';')[1].replace("\'","\"")
        ast = ast.replace("False", "false")
        ast = ast.replace("True", "true")
        #ast = demjson.decode(ast)
        ast = json.loads(ast)
        hole_index = int(hole_index.split(';')[1])
        ori_pred = ori_pred.split(';')[1]
        embed_pred = embed_pred.split(';')[1]
        yield ast, hole_index, ori_pred, embed_pred
        ast = file.readline()
        #return ast, hole_index, ori_pred, embed_pred




def get_string(ast, hole_index, is_terminal):
    """给定一个AST，使用递归方式将其转换成一个string file"""
    def ast2code(token):
        if type(token) == int:
            return
        return_string = ''
        type_info = token['type']
        if token['id'] == hole_index and is_terminal:
            return_string += ' ___________ '
            return return_string

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
                 #+ str(ast[token['children'][0]]) + str(ast[token['children'][1]])
                if len(child_list) == 1:
                    return_string += 'function () ' + ast2code(ast[child_list[0]])
                else:
                    for i, child in enumerate(child_list):
                        if i == 0:
                            return_string += 'function (' + ast2code(ast[child_list[0]]) + ')'
                        else:
                            return_string += ast2code(ast[child_list[i]])

            elif type_info == 'AssignmentPattern':
                raise KeyError('There is no non-terminal token: {}'.format(token))
            elif type_info == 'SequenceExpression':
                for child in child_list:
                    return_string += ast2code(ast[child]) + ', '

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
                if 'value' in token:
                    return_string += str(token['value']) + ':'
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
            if 'value' in token:
                return_string += token['value'] + ', '
            else:
                return_string += ' '
        elif type_info == 'BlockStatement':
            return_string += "{}"
        elif type_info == 'ObjectExpression':
            return_string += ''
        elif type_info == 'ThisExpression':
            return_string += 'this'
        elif type_info == 'ArrayExpression':
            return_string += ""
        elif type_info == 'ReturnStatement':
            return_string +='return'

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
    elif type_info == 'LiteralRegExp':
        return token['value']

    else:
        print('error', token)
        raise KeyError('terminal error', str(token))


def pretty_print(cate):
    """Using indent commend to pretty the source code"""
    if cate == 'ori':
        dir_path = 'temp_data/predict_compare/compare_source_code/ori_correct/'
    elif cate == 'embed':
        dir_path = 'temp_data/predict_compare/compare_source_code/embed_correct/'
    elif cate == 'both':
        dir_path = 'temp_data/predict_compare/compare_source_code/both_wrong/'
    else:
        raise KeyError('No this category')
    file_list = os.listdir(dir_path)
    for i, file in enumerate(file_list):
        if file.startswith('.'):
            continue
        #subprocess.check_output()
        commend = 'indent ' + dir_path + file
        os.system(commend)
        print('beautify', file)


def all_ast_to_string():
    # 读取所有compare的ast并转换成source code
    ori_corr_file_index = 1
    embed_corr_file_index = 1
    both_wrong_file_index = 1
    for ast, expect_index, ori_pred, embed_pred, expect_token in get_test_ast_with_pickle(is_terminal=True):
        if len(ast) > 3000 or len(ast) < 1:
            continue
        string_code = get_string(ast, expect_index, is_terminal=True)
        if ori_pred != expect_token and embed_pred != expect_token:
            path = 'temp_data/predict_compare/compare_source_code/both_wrong/code%d.txt' % (both_wrong_file_index)
            both_wrong_file_index += 1
        elif ori_pred != expect_token:
            path = 'temp_data/predict_compare/compare_source_code/embed_correct/code%d.txt' % (embed_corr_file_index)
            embed_corr_file_index += 1
        elif embed_pred != expect_token:
            path = 'temp_data/predict_compare/compare_source_code/ori_correct/code%d.txt' % (ori_corr_file_index)
            ori_corr_file_index += 1
        else:
            raise KeyError('all predict correct')
        with open(path, 'w') as file:
            file.write(string_code)
            file.write('\n\n\n\n')
            file.write('ori_predict: ')
            file.write(ori_pred)
            file.write('\n')
            file.write('embed_predict: ')
            file.write(embed_pred)
            file.write('\n')
            file.write('expect_token: ')
            file.write(expect_token)
            file.write('\n')


if __name__ == '__main__':

    # 测试用，使用js parser将helloworld.js文件处理成AST，并再将其转换成string code
    # ast = get_json()
    # for i in ast:
    #     print(i)
    # string = get_string(ast, -1, is_terminal=False)
    # print(string)
    #
    # file = open('temp_string_js_code.text', 'w')
    # file.write(string)
    # file.close()

    # 读取所有compare的ast并转换成source code
    #all_ast_to_string()

    # 使用indent对指定路径文件进行beautify
    pretty_print(cate='both')

