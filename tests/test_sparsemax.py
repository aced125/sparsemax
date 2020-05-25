#!/usr/bin/env python

"""Tests for `sparsemax` package."""

import pytest
from sparsemax import Sparsemax
from torch.autograd import gradcheck
import torch


@pytest.mark.parametrize("dimension", [-4, -3, -2, -1, 0, 1, 2, 3])
def test_sparsemax(dimension):
    sparsemax = Sparsemax(dimension)
    input = torch.randn(6, 3, 5, 4, dtype=torch.double, requires_grad=True)
    assert gradcheck(sparsemax, input, eps=1e-6, atol=1e-4)


def test_sparsemax_invalid_dimension():
    sparsemax = Sparsemax(-7)
    input = torch.randn(6, 3, 5, 4, dtype=torch.double, requires_grad=True)
    with pytest.raises(IndexError):
        gradcheck(sparsemax, input, eps=1e-6, atol=1e-4)
