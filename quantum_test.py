"""Tests for Simple Quantum Simulations."""

import collections
import functools
import operator
import pytest
import numpy as np
import quantum


TRIALS = 100000


def test_qubit_enforce_normalized():
    state = [0, 2]
    with pytest.raises(ValueError):
        quantum.Qubit(state)


def test_qubits_enforce_normalized():
    state = [0, 1, 1, 1]
    with pytest.raises(ValueError):
        quantum.Qubits(state)


def test_num_qubits():
    possible_qubits = (quantum.zero(),
                       quantum.one(),
                       quantum.super_position(quantum.zero()))
    for i in range(2, 5):
        qubits = [np.random.choice(possible_qubits) for _ in range(i)]
        qubits = quantum.tensor_product(*qubits)
        assert qubits.num_qubits == i


def test_make_entangled():
    qus = quantum.make_entangled()
    np.testing.assert_allclose(qus.state, np.array([quantum._ONE_OVER_ROOT_TWO,
                                                    0,
                                                    0,
                                                    quantum._ONE_OVER_ROOT_TWO]))


def test_entangled():
    for _ in range(TRIALS):
        entangled = quantum.make_entangled()
        measured = entangled.measure()
        q1, q2 = quantum.tensor_factor(measured)
        np.testing.assert_allclose(q1.state, q2.state)


@pytest.mark.parametrize("i,expected",
    ((quantum.zero(), np.array([quantum._ONE_OVER_ROOT_TWO,
                                quantum._ONE_OVER_ROOT_TWO])),
     (quantum.one(), np.array([quantum._ONE_OVER_ROOT_TWO,
                               -quantum._ONE_OVER_ROOT_TWO]))))
def test_hadamard(i, expected):
    q = quantum.super_position(i)
    np.testing.assert_allclose(q.state, expected)


@pytest.mark.parametrize("func", (quantum.super_position, quantum.negation))
@pytest.mark.parametrize("i", (quantum.zero(), quantum.one()))
def test_inverse(func, i):
    np.testing.assert_allclose(func(func(i)).state, i.state, atol=1e-7)


@pytest.mark.parametrize("i,expected", ((quantum.zero(), quantum.one()),
                                        (quantum.one(), quantum.zero())))
def test_negation(i, expected):
    np.testing.assert_allclose(quantum.negation(i).state, expected.state)


def test_super_position_random():
    counts = collections.Counter()
    for _ in range(TRIALS):
        q = quantum.super_position(quantum.zero())
        q = q.measure()
        state = np.argmax(q.state)
        counts[state] += 1
    norm = sum(counts.values())
    probs = {k: v / norm for k, v in counts.items()}
    for v in probs.values():
        np.testing.assert_allclose(v, 0.5, rtol=1e-1)


@pytest.mark.parametrize("qubits,expected",
    (((quantum.zero(), quantum.zero()), np.array([1, 0, 0, 0])),
     ((quantum.zero(), quantum.one()), np.array([0, 1, 0, 0])),
     ((quantum.one(), quantum.zero()), np.array([0, 0, 1, 0])),
     ((quantum.one(), quantum.one()), np.array([0, 0, 0, 1])),
     ((quantum.one(), quantum.one(), quantum.zero()), np.array([0, 0, 0, 0, 0, 0, 1, 0])),
     ((quantum.one(), quantum.zero(), quantum.zero()), np.array([0, 0, 0, 0, 1, 0, 0, 0]))))
def test_tensor_product(qubits, expected):
    qs = quantum.tensor_product(*qubits)
    np.testing.assert_allclose(qs.state, expected)
    qs2 = functools.reduce(operator.matmul, qubits)
    np.testing.assert_allclose(qs2.state, qs.state)


@pytest.mark.parametrize("func", (quantum.constant_0,
                                  quantum.constant_1,
                                  quantum.identity,
                                  quantum.negation_two_op))
@pytest.mark.parametrize("i", (quantum.zero(), quantum.one()))
def test_two_op_inverse(func, i):
    o = quantum.zero()
    i2, o2 = func(i, o)
    i2, o2 = func(i2, o2)
    np.testing.assert_allclose(i2.state, i.state)
    np.testing.assert_allclose(o2.state, o.state)


@pytest.mark.parametrize("i", (quantum.one(), quantum.zero()))
def test_constant_0(i):
    o = quantum.zero()
    i2, o2 = quantum.constant_0(i, o)
    np.testing.assert_allclose(i2.state, i.state)
    np.testing.assert_allclose(o2.state, quantum.zero().state)


@pytest.mark.parametrize("i", (quantum.one(), quantum.zero()))
def test_constant_1(i):
    o = quantum.zero()
    i2, o2 = quantum.constant_1(i, o)
    np.testing.assert_allclose(i2.state, i.state)
    np.testing.assert_allclose(o2.state, quantum.one().state)


@pytest.mark.parametrize("control,input,expected",
    ((quantum.zero(), quantum.zero(), quantum.zero()),
     (quantum.zero(), quantum.one(), quantum.one()),
     (quantum.one(), quantum.zero(), quantum.one()),
     (quantum.one(), quantum.one(), quantum.zero())))
def test_cnot(control, input, expected):
    control2, output = quantum.cnot(control, input)
    np.testing.assert_allclose(control2.state, control.state)
    np.testing.assert_allclose(output.state, expected.state)


@pytest.mark.parametrize("i", (quantum.one(), quantum.zero()))
def test_identity(i):
    o = quantum.zero()
    i2, o2 = quantum.identity(i, o)
    np.testing.assert_allclose(i2.state, i.state)
    np.testing.assert_allclose(o2.state, i.state)


@pytest.mark.parametrize("i,expected",
                          ((quantum.one(), quantum.zero()),
                           (quantum.zero(), quantum.one())))
def test_negation(i, expected):
    o = quantum.zero()
    i2, o2 = quantum.negation_two_op(i, o)
    np.testing.assert_allclose(i2.state, i.state)
    np.testing.assert_allclose(o2.state, expected.state)


@pytest.mark.parametrize("func,expected", ((quantum.identity, "variable"),
                                           (quantum.negation_two_op, "variable"),
                                           (quantum.constant_0, "constant"),
                                           (quantum.constant_1, "constant")))
def deutsch_oracle(func, expected):
    assert func() == expected
