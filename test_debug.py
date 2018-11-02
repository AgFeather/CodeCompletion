import pickle

js_test_data_dir = 'js_dataset/js_programs_eval.json'
js_train_data_dir = 'js_dataset/js_programs_training.json'


def dataset_split(is_training=True, subset_size=5000):
    '''
    read orginal 11G JavaScript source code(AST format), and seperate the whole data into several subdataset
    and saved with pickle.
    :param is_training:
    :param subset_size:
    :return:
    '''
    if is_training:
        data_path = js_train_data_dir
        total_size = 100000
        saved_to_path = 'split_js_data/train_data/'
    else:
        data_path = js_test_data_dir
        total_size = 50000
        saved_to_path = 'split_js_data/eval_data/'

    file = open(data_path, 'r')
    subset_list = []
    error_count = 0
    for i in range(1, total_size + 1):
        try:
            line = file.readline()
        except:
            error_count += 1
           # print('UTF-8 error: {}/{}'.format(error_count, i))
        subset_list.append(line)
        if i % subset_size == 0:
            sub_path = saved_to_path + 'part{}'.format(i // subset_size) + '.json'
            save_file = open(sub_path, 'wb')
            pickle.dump(subset_list, save_file)
            subset_list = []
    print('data seperating finished..., utf-8 error:{}'.format(error_count))





example = [{'id': 0, 'child': [1, 2, 3]},
           {'id': 1, 'child': [4, 5, 6]}, {'id': 2, 'child': [7]}, {'id': 3, 'child': [8, 9]},
           {'id': 4}, {'id': 5}, {'id': 6}, {'id': 7}, {'id': 8}, {'id': 9}]


def bulid_binary_tree(data):
    '''
    transform the AST(one node may has many childNodes) to Left-Child-Right-Sibilng(LCRS) binary tree.
    :param data:
    :return:
    '''
    brother_map = {1: []}
    for node in data:
        node['right'] = brother_map.get(node['id'], [])
        if 'child' in node.keys():
            brother_map[node['child'][0]] = node['child'][1:]
            node['left'] = node['child'][0]
    return data


def AST_to_seq(data):
    bi_tree = bulid_binary_tree(data)
    AST_seq = []
    def in_order_traversal(data, index):
        if 'left' in data[index].keys():
            in_order_traversal(data, data[index]['left'])
        brother_list = data[index]['right']
        AST_seq.append(data[index])
        for node_index in brother_list:
            in_order_traversal(data, node_index)
    in_order_traversal(bi_tree, 0)
    return AST_seq



if __name__ == '__main__':
    ast_seq = AST_to_seq(example)
    print(ast_seq)


