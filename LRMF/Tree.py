from collections import defaultdict
import numpy as np

class Node(object):
    def __init__(self, users, question, like=None, dislike=None, child=None):
        self.users = users                      # Set of users
        self.question = question                # item id used for question
        self.like = like                        # Like child
        self.dislike = dislike                  # Dislike child
        self.child = child                      # If no split can be performed
        self.flike = None
        self.like_group = None

    def is_leaf(self):
        if self.like is None and self.dislike is None and self.child is None:
            return True
        return False

    def _questions(self):
        if self.is_leaf():
            return [[]]  # one path: only contains self.value
        paths = []
        for child in [self.like, self.dislike]:
            for path in child._questions():
                paths.append([self.question] + path)
        return paths

    def _local_questions(self):
        if self.is_leaf():
            return [self.local_questions]
        return self.like._groups() + self.dislike._groups()

    def _groups(self):
        if self.is_leaf():
            return [self.users]
        return self.like._groups() + self.dislike._groups()

    def set_transformation(self, transformation):
        self.transformation = transformation

    def set_globals(self, globals):
        self.global_questions = globals

    def set_locals(self, locals):
        self.local_questions = locals


def traverse_a_user(user: int, data, tree: Node):
    if tree.is_leaf():
        return tree

    if tree.child == None:
        if data[tree.question] >= 1: # This line needs to be changed for some models.
            return traverse_a_user(user, data, tree.like)
        else:
            return traverse_a_user(user, data, tree.dislike)
    else:
        return traverse_a_user(user, data, tree.child)

def traverse_a_user_eatnn(user: int, data, tree: Node):
    if tree.is_leaf():
        return tree

    if tree.child == None:
        if data[user, tree.question] >= 1: # This line needs to be changed for some models.
            return traverse_a_user_eatnn(user, data, tree.like)
        else:
            return traverse_a_user_eatnn(user, data, tree.dislike)
    else:
        return traverse_a_user_eatnn(user, data, tree.child)



def traverse_a_user_social(user: int, data, social, tree: Node):
    if tree.is_leaf():
        return tree

    if tree.child == None:
        friends_who_like = (set(social[social.uid == user].sid)).intersection(set(tree.like_group))
        if data[tree.question] >= 1: # This line needs to be changed for some models.
            return traverse_a_user_social(user, data, social, tree.like)
        elif friends_who_like:
            return traverse_a_user_social(user, data, social, tree.flike)
        else:
            return traverse_a_user_social(user, data, social, tree.dislike)
    else:
        return traverse_a_user_social(user, data, social, tree.child)
