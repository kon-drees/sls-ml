from enum import Enum
from typing import List, Tuple

import numpy as np


class Label(Enum):
    IN = 1
    OUT = 0
    UNDEC = -1


class ArgumentationFramework:
    def __init__(self):
        self.args = None
        self.arguments = set()
        self.attacks = set()

    def add_argument(self, arg):
        self.arguments.add(arg)

    def add_attack(self, attacker, target):
        self.attacks.add((attacker, target))

    def add_arguments_from(self, arguments: list[str]):
        for argument in arguments:
            self.add_argument(argument)

    def add_attacks_from(self, attacks: list[tuple[str, str]]):
        for attack in attacks:
            self.add_attack(attack[0], attack[1])

    def remove_argument(self, arg):
        self.arguments.discard(arg)
        self.attacks = {(attacker, target) for attacker, target in self.attacks if attacker != arg and target != arg}

    def remove_attack(self, attacker, target):
        self.attacks.discard((attacker, target))

    def get_attacks_from(self, arg):
        return {target for attacker, target in self.attacks if attacker == arg}

    def get_attacks_to(self, arg):
        return {attacker for attacker, target in self.attacks if target == arg}

    def get_neighbors(self, arg):
        return self.get_attacks_from(arg) | self.get_attacks_to(arg)

    def get_arguments(self):
        return self.arguments

    def get_attacks(self):
        return self.attacks

    def get_number_of_attacks(self):
        return len(self.get_attacks())

    def get_number_of_arguments(self):
        return len(self.get_arguments())

    def in_degree(self, arg):
        return len(self.get_attacks_to(arg))

    def out_degree(self, arg):
        return len(self.get_attacks_from(arg))

    def get_most_attacked_argument(self):
        return max(self.args, key=lambda arg: len(self.get_attacks_from(arg)))

    def get_most_attacking_argument(self):
        return max(self.args, key=lambda arg: len(self.get_attacks_to(arg)))

    def adjacency_matrix(self):
        nodes = list(self.arguments)
        node_indices = {node: i for i, node in enumerate(nodes)}
        matrix = np.zeros((len(nodes), len(nodes)), dtype=int)

        for attacker, target in self.attacks:
            matrix[node_indices[attacker], node_indices[target]] = 1

        return matrix

    def is_attacked_by(self, arg1, arg2):
        return (arg2, arg1) in self.attacks

    def is_empty(self):
        return len(self.arguments) == 0

    def to_tgf(self):
        node_lines = "\n".join(str(arg) for arg in self.arguments)
        edge_lines = "\n".join(f"{attacker} {target}" for attacker, target in self.attacks)
        return f"{node_lines}\n#\n{edge_lines}"

    def to_apx(self):
        arg_lines = "\n".join(f"arg({arg})." for arg in self.arguments)
        attack_lines = "\n".join(f"att({attacker}, {target})." for attacker, target in self.attacks)
        return f"{arg_lines}\n{attack_lines}"

    def __len__(self):
        return len(self.arguments)

    def __str__(self):
        arg_str = ", ".join(map(str, self.arguments))
        attack_str = ", ".join(f"({attacker}, {target})" for attacker, target in self.attacks)
        return f"Arguments: {{{arg_str}}}\nAttacks: {{{attack_str}}}"
