=========
sparsemax
=========


.. image:: https://img.shields.io/pypi/v/sparsemax.svg
        :target: https://pypi.python.org/pypi/sparsemax

.. image:: https://img.shields.io/travis/aced125/sparsemax.svg
        :target: https://travis-ci.com/aced125/sparsemax

.. image:: https://readthedocs.org/projects/sparsemax/badge/?version=latest
        :target: https://sparsemax.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/aced125/sparsemax/shield.svg
     :target: https://pyup.io/repos/github/aced125/sparsemax/
     :alt: Updates

.. image:: coverage.svg




A PyTorch implementation of SparseMax (https://arxiv.org/pdf/1602.02068.pdf) with gradients checked and tested

Sparsemax is an alternative to softmax when one wants to generate
hard probability distributions. It has been used to great effect in recent papers like
ProtoAttend (https://arxiv.org/pdf/1902.06292v4.pdf).

Installation
------------

.. code-block:: bash

   pip install -U sparsemax


Usage
-----

Use as if it was :code:`nn.Softmax()`! Nice and simple.

.. code-block:: python

    from sparsemax import Sparsemax
    import torch
    import torch.nn as nn

    sparsemax = Sparsemax(dim=-1)
    softmax = torch.nn.Softmax(dim=-1)

    logits = torch.randn(2, 3, 5)
    logits.requires_grad = True
    print("\nLogits")
    print(logits)

    softmax_probs = softmax(logits)
    print("\nSoftmax probabilities")
    print(softmax_probs)

    sparsemax_probs = sparsemax(logits)
    print("\nSparsemax probabilities")
    print(sparsemax_probs)


Advantages over existing implementations
----------------------------------------
This repo borrows heavily from: https://github.com/KrisKorrel/sparsemax-pytorch

However, there are a few key advantages:

1. Backward pass equations implemented natively as a :code:`torch.autograd.Function`, **resulting in 30% speedup**, compared to the above repository.
2. The package is **easily pip-installable** (no need to copy the code).
3. The package works for **multi-dimensional tensors, operating over any axis**.
4. The operator **forward and backward passes are tested** (backward-pass check due to :code:`torch.autograd.gradcheck`


Check that gradients are computed correctly
-------------------------------------------

.. code-block:: python

    from torch.autograd import gradcheck
    from sparsemax import Sparsemax

    input = (torch.randn(6, 3, 20,dtype=torch.double,requires_grad=True))
    test = gradcheck(sparsemax, input, eps=1e-6, atol=1e-4)
    print(test)



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
