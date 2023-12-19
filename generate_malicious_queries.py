from nltk import word_tokenize
from nltk.parse.stanford import StanfordParser
import os
import math
from nltk.tree import Tree

# download necessary dataset(first time)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# set the stanford path
os.environ['STANFORD_PARSER'] = 'stanford-parser'
os.environ['STANFORD_MODELS'] = 'stanford-parser'

# set top N
N = 3


def query_model(sentence):
    # TODO you should query the distilled model to get the sentence score
    return 0


def calculate_importance_score(full_sentence, sub_tree):
    full_semantics_score = query_model(full_sentence)

    sub_sentence = " ".join(sub_tree.leaves())
    sub_semantics_score = query_model(full_sentence.replace(sub_sentence, ""))
    return math.fabs((full_semantics_score - sub_semantics_score) / len(sub_sentence))


def traverse_tree(tree, subtrees=None):
    if subtrees is None:
        subtrees = []

    for subtree in tree:
        if isinstance(subtree, Tree):
            # traverse the tree and collect all sub syntax trees
            subtrees.append(subtree)
            traverse_tree(subtree, subtrees)
    return subtrees


def gen_syntax_tree(sentence):
    tokens = word_tokenize(sentence)
    parser = StanfordParser(model_path="englishPCFG.ser.gz")

    # generate syntax tree
    trees = list(parser.parse(tokens))
    return trees


def parse_syntax_info(sentence):
    s_tree = gen_syntax_tree(sentence)
    s_sub_trees = traverse_tree(s_tree)
    s_importance_scores = []

    for s_sub_tree in s_sub_trees:
        s_importance_scores.append(calculate_importance_score(s1, s_sub_tree))
    # get a list with sub syntax tree and its importance score
    s_tree_info = list(zip(s_sub_trees, s_importance_scores))
    s_tree_info.sort(key=get_score, reverse=True)

    return s_tree_info


def get_score(element):
    return element[1]


if __name__ == '__main__':
    s1 = "Malicious query1"
    s1_syntax_info = parse_syntax_info(s1)

    s2 = "Malicious query2"
    s2_syntax_info = parse_syntax_info(s2)

    new_s = []
    # replace candidate sub syntax tree
    for i in range(N):
        candidate_tree = s1_syntax_info[i][0]
        for sub_tree, score in s2_syntax_info:
            if sub_tree.label() == candidate_tree.label():
                new_s.append(s2.replace(" ".join(sub_tree.leaves()), " ".join(candidate_tree.leaves())))
                break

    print(new_s)
