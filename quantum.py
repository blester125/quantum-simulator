"""Simple simulation of Quantum Computing with the Quantum Gate Model.

"""

import dataclasses
import itertools
from typing import Tuple, Callable, Union
import numpy as np


@dataclasses.dataclass
class Qubit:
    """A Qubit, our base unit of computation.

    Attributes:
        state: The state of the qubit.

    The state of qubit is normalized, that is, given a state of [a, b] then
    ǁaǁ² + ǁbǁ² = 1.

    The probability of the Qubit being in any state is the square of the state.

    Note:
        The qubit is not secretly in one state or the other until it is
        measured, it is in both states at once. This is called "super position"
    """

    state: np.ndarray

    def __post_init__(self):
        self.state = np.array(self.state)
        # Make sure the qubit is normalized.
        if not np.allclose(np.sum(np.square(self.state)), 1.0):
            raise ValueError("Qubit state is not normalized!")

    def __repr__(self):
        return f"{self.__class__.__name__}(state={self.state!r})"

    def __str__(self):
        if self.is_zero():
            return "|0⟩"
        if self.is_one():
            return "|1⟩"
        state = self.state.tolist()
        state = [format_one_over_root_2(s) for s in state]
        state = ", ".join(state)
        return f"{self.__class__.__name__}(state=[{state}])"

    def __matmul__(self: 'Qubit', other: 'Qubit') -> 'Qubits':
        return tensor_product(self, other)

    def probability(self) -> np.ndarray:
        """The probability of being in any state is the square of the state."""
        return np.square(self.state)

    def measure(self):
        """Collapse the wave function, results in a Qubit of either |0⟩ or |1⟩."""
        m = np.random.choice(range(len(self.state)), p=np.square(self.state))
        state = np.zeros_like(self.state)
        state[m] = 1
        return self.__class__(state=state)

    def is_one(self) -> bool:
        """Does the Qubit represent a classical 1 bit?"""
        return abs(self.state[1]) == 1

    def is_zero(self) -> bool:
        """Does the Qubit represent a classical 0 bit?"""
        return abs(self.state[0]) == 1

    def __neg__(self):
        return negation(self)


def zero() -> Qubit:
    """Create a Qubit with the value of |0⟩."""
    return Qubit([1, 0])


def one() -> Qubit:
    """Create a Qubit with a value of |1⟩."""
    return Qubit([0, 1])


@dataclasses.dataclass
class Qubits(Qubit):
    """A class holding multiple Qubits. We might be able to remove this."""
    state: np.ndarray

    @property
    def num_qubits(self):
        """How many qubits are represented by this state."""
        return int(np.log2(len(self.state)))

    def entangled(self) -> bool:
        qubits = tensor_factor(self)
        if isinstance(qubits, Qubits):
            return True
        return False

    def __repr__(self):
        return f"{self.__class__.__name__}(state={self.state!r})"

    def __matmul__(self: 'Qubits', other: 'Qubit') -> 'Qubits':
        reshape = tuple([2 for _ in range(self.num_qubits)])
        value = np.reshape(self.state, reshape)
        return Qubits(state=np.tensordot(value,
                                         other.state,
                                         axes=0).ravel())

    def __str__(self):
        if self.entangled():
            state = self.state.tolist()
            state = [format_one_over_root_2(s) for s in state]
            state = ", ".join(state)
            return f"Entangled{self.__class__.__name__}(state=[{state}])"
        qubits = tensor_factor(self)
        qubits = " ⊗ ".join(str(q) for q in qubits)
        return f"{self.__class__.__name__}(state={qubits})"


def format_one_over_root_2(v):
    if np.allclose(abs(v), _ONE_OVER_ROOT_TWO):
        if v < 0:
            return "-1/√2"
        else:
            return "1/√2"
    return str(v)


def tensor_product(*qubits: Qubit) -> Qubits:
    """Create a vector representaiton of multiple qubits with the Knocker Product."""
    # This can also be done with a ⊗ b = np.reshape((b @ a.T), (-1, 1))
    current = qubits[0].state
    for qubit in qubits[1:]:
        current = np.tensordot(current, qubit.state, axes=0)
    return Qubits(current.ravel())


# Various constants used for Qubit manipulations.
_IDENTITY = np.array([[1, 0],
                      [0, 1]])
_NEGATION = np.array([[0, 1],
                      [1, 0]])
_CONSTANT_0 = np.array([[1, 1],
                        [0, 0]])
_CONSTANT_1 = np.array([[0, 0],
                        [1, 1,]])
_CNOT = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]])
_ONE_OVER_ROOT_TWO = 1 / np.sqrt(2)
_HADAMARD = np.array([[_ONE_OVER_ROOT_TWO, _ONE_OVER_ROOT_TWO],
                      [_ONE_OVER_ROOT_TWO, -_ONE_OVER_ROOT_TWO]])


def _make_factorization_table():
    """Create a table of products to unit-circle Qubits."""
    unit_circle_qubits = (
        Qubit([0, 1]),
        Qubit([_ONE_OVER_ROOT_TWO, _ONE_OVER_ROOT_TWO]),
        Qubit([1, 0]),
        Qubit([_ONE_OVER_ROOT_TWO, -_ONE_OVER_ROOT_TWO]),
        Qubit([0, -1]),
        Qubit([-_ONE_OVER_ROOT_TWO, -_ONE_OVER_ROOT_TWO]),
        Qubit([-1, 0]),
        Qubit([-_ONE_OVER_ROOT_TWO, _ONE_OVER_ROOT_TWO]),
    )
    factorization_table = {}
    for qubits in itertools.product(unit_circle_qubits,
                                    unit_circle_qubits):
        # Only save the first value so it we tend to stay positive.
        state = tuple(tensor_product(*qubits).state.tolist())
        if state not in factorization_table:
            factorization_table[state] = tuple(tuple(q.state) for q in qubits)
    for qubits in itertools.product(unit_circle_qubits,
                                    unit_circle_qubits,
                                    unit_circle_qubits):
        # Only save the first value so it we tend to stay positive.
        state = tuple(tensor_product(*qubits).state.tolist())
        if state not in factorization_table:
            factorization_table[state] = tuple(tuple(q.state) for q in qubits)
    return factorization_table


_FACTORIZATION_TABLE = _make_factorization_table()


def tensor_factor(qubits: Qubits) -> Union[Tuple[Qubit], Qubits]:
    """Factor a Tensor Product to Qubits.

    Note:
        We use a table of common products to avoid a general factoring for now.
    """
    try:
        states = _FACTORIZATION_TABLE[tuple(qubits.state.tolist())]
        return tuple(Qubit(s) for s in states)
    except KeyError:
        return qubits


def _single_bit_op(qubit: Qubit, transform: np.ndarray) -> Qubit:
    """Apply a function, represented by `transform` to `qubit`."""
    return Qubit(state=qubit.state @ transform)


def negation(qubit: Qubit) -> Qubit:
    """Flip a qubit."""
    return _single_bit_op(qubit, _NEGATION)


def cnot(control: Qubit, input: Qubit) -> Tuple[Qubit, Qubit]:
    """Flip the `input` Qubit iff the `control` Qubit is 1."""
    qubits = tensor_product(control, input)
    result = qubits.state @ _CNOT
    return tensor_factor(Qubits(result))


def super_position(qubit: Qubit) -> Qubit:
    """Move the qubit into an even super_position of |0⟩ and |1⟩."""
    return _single_bit_op(qubit, _HADAMARD)


def make_entangled() -> Qubits:
    """Create two Qubits that are entangled."""
    q1 = zero()
    q2 = zero()
    q1 = super_position(q1)
    return cnot(q1, q2)


def negation_two_op(input: Qubit, output: Qubit) -> Tuple[Qubit, Qubit]:
    """Store the negation of the `input` qubit in the `output` qubit."""
    input, output = cnot(input, output)
    return input, negation(output)


def identity(input: Qubit, output: Qubit) -> Tuple[Qubit, Qubit]:
    """Store the `input` qubit in the `output` qubit."""
    return cnot(input, output)


def constant_0(input: Qubit, output: Qubit) -> Tuple[Qubit, Qubit]:
    """Store |0⟩ in the `output` qubit."""
    return input, output


def constant_1(input: Qubit, output: Qubit) -> Tuple[Qubit, Qubit]:
    """Store |1⟩ in the `output` qubit."""
    return input, negation(output)


def deutsch_oracle(func: Callable[[Qubit, Qubit], Tuple[Qubit, Qubit]]) -> bool:
    """Determin if `func` is a constant or variable function in 1 query."""
    input = zero()
    output = zero()
    input = negation(input)
    output = negation(output)
    input = super_position(input)
    output = super_position(output)
    input, output = func(input, output)
    input = super_position(input)
    output = super_position(output)
    input = input.measure()
    output = output.measure()
    if input.is_one() and output.is_one():
        return "constant"
    if input.is_zero() and output.is_one():
        return "variable"


if __name__ == "__main__":
    print(f"Zero: {zero()}")
    print(f"One: {one()}")
    print(f"Super Position: {super_position(zero())}")
    qs = tensor_product(one(), zero(), super_position(zero()))
    print(f"MultiQubit: {qs!r} = {qs}")
    print(f"Identity is '{deutsch_oracle(identity)}'.")
    print(f"Negation is '{deutsch_oracle(negation_two_op)}'.")
    print(f"Constant 0 is '{deutsch_oracle(constant_0)}'.")
    print(f"Constant 1 is '{deutsch_oracle(constant_1)}'.")
